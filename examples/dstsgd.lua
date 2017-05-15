local ipc = require 'libipc'
local threads = require 'threads'
local posix = require 'posix'

-- nodes is a list of hostname and port numbers
-- node_weights is a matrix for parameter weights for different nodes
-- node_id is the ID of this node
-- self_parameters is a table of tensors of training parameters
local function DecentralizedSGD(nodes, node_weights, node_id, model_parameters, cuda, chunk_size, early_average, gpu_buffer_storage, input_weights_storage)

  -- use cuda or not
  if cuda == nil then
    cuda = true
  end
  if early_average == nil then
    early_average = false
  end
  local copy_from_gpu = false
  local gpu_buffer
  local gpu_input_weights
  if gpu_buffer_storage ~= nil then
    copy_from_gpu = true
    cuda = true
    require 'cutorch'
    gpu_buffer = torch.CudaTensor(gpu_buffer_storage)
    gpu_input_weights = torch.CudaTensor(input_weights_storage)
  end
  -- thread pool
  local pool
  -- network parameters of myself
  local self_ip, self_port, self_rank, self_weight
  -- weights for all connected peers
  local peer_weights
  -- number of incoming peers
  local nr_incoming
  -- total number of worker threads
  local total_clients
  -- shared flag for exiting worker threads
  local exit_flag = torch.IntTensor(1):fill(0)
  -- thread pool for clients and servers
  local pool
  -- threads for clients
  local clients = { }
  -- weights for connected clients
  local clients_weights = { }
  local n_clients_weights
  -- share upvalue tensors between threads
  threads.serialization('threads.sharedserialize')
  -- mutex and conditional variable for iteration synchronization between main thread and communication threads
  local server_sync_lock = threads.Mutex()
  local server_sync_lock_id = server_sync_lock:id()
  local server_lock = threads.Mutex()
  local server_lock_id = server_lock:id()
  local server_sync_cond = threads.Condition()
  local server_sync_cond_id = server_sync_cond:id()
  local client_sync_lock = threads.Mutex()
  local client_sync_lock_id = client_sync_lock:id()
  local client_lock = threads.Mutex()
  local client_lock_id = client_lock:id()
  local client_sync_cond = threads.Condition()
  local client_sync_cond_id = client_sync_cond:id()
  -- the counter will +1 after send/get one weight
  local client_sync_progress = torch.IntTensor(1):fill(0)
  local server_sync_progress = torch.IntTensor(1):fill(0)
  -- create tensors for receiving tensors from peers
  local t_recv = { } -- a table of torch.Tensor() for receiving tensors
  -- flatten the parameter table
  local self_parameters = { }
  -- chunk size
  chunk_size = chunk_size or 16384
  -- debug print's
  local debug = false
  local thread_print = print
  
  local function ExpandParameters(param, keyname)
    if type(param) == 'table' then
      for key, val in pairs(param) do
        ExpandParameters(val, keyname..'_'..key)
      end
    elseif string.match(torch.typename(param), "Tensor") then
      local prev_type = torch.getdefaulttensortype()
      torch.setdefaulttensortype(param:type())
      -- flatten the tensor to 1D
      self_parameters[keyname] = torch.Tensor(param:storage())
      torch.setdefaulttensortype(prev_type)
    end
  end

  -- find IP of this node
  local function GetSelfIP()
    if node_id == 0 then
      -- If nodeID is not given, we will guess our own IP and find it in the table
      require "socket"
      local s = socket.udp()
      s:setpeername("8.8.8.8",53)
      local ip, _ = s:getsockname()
      local found = false
      for rank, hostport in ipairs(nodes) do
        if hostport["host"] == ip or socket.dns.toip(hostport["host"]) == ip or hostport["host"] == socket.dns.gethostname() then
          found = true
          hostport["self"] = true
          self_ip = ip
          self_port = hostport["port"]
          self_rank = rank
          break
        end
      end
      if not found then
        error('Cannot found self IP in nodes list. Try to set nodeID.')
        return
      end
    else
      self_rank = node_id
      self_ip = nodes[node_id]["host"]
      self_port = nodes[node_id]["port"]
      nodes[self_rank]["self"] = true
    end
    print(string.format("This machine is %s:%s, rank %d", self_ip, self_port, self_rank))
  end

  -- get weights for this node and all peers 
  local function GetSelfWeight()
    -- nr_incoming is the number of incoming peers
    nr_incoming = 0
    for i, weights in pairs(node_weights) do
      -- add the counter nr_incoming when some peer needs to connect to me
      if tonumber(weights[self_rank]) ~= 0 then
        nr_incoming = nr_incoming + 1
      end
      -- the row for this node
      if (i == self_rank) then
        -- remove ourself from clients
        nr_incoming = nr_incoming - 1
        peer_weights = tablex.copy(weights)
        self_weight = peer_weights[self_rank]
        peer_weights[self_rank] = 0.0
      end
    end
    print(self_weight)
    print(peer_weights)
    print(nr_incoming)
  end

  -- define server thread
  local server_thread = function(server_sync_lock_id, server_sync_cond_id, server_lock_id, client_lock_id) 
    local ipc = require 'libipc'
    local threads = require 'threads'
    local posix = require 'posix'
    local server_sync_lock = threads.Mutex(server_sync_lock_id)
    local server_sync_cond = threads.Condition(server_sync_cond_id)
    local server_lock = threads.Mutex(server_lock_id)
    local client_lock = threads.Mutex(client_lock_id)
    thread_print = function(str)
      if debug then
        local colors = require 'ansicolors'
        local sec, nano
        sec, nano = posix.clock_gettime('CLOCK_MONOTONIC')
        print(colors(string.format('%%{1}[%10d.%9d][SERVER 1/1 %d] %s', sec, nano, __threadid, str)))
      end
    end
    thread_print(string.format("mutex ID is %d", server_sync_lock_id))
    -- initialize connection
    local server = ipc.server(self_ip, self_port)
    -- on initialization, wait for num_clients clients
    thread_print(string.format('Server at %s:%d waiting...', self_ip, self_port))
    -- generate sorted keys
    local ordered_keys = {}
    for k in pairs(self_parameters) do
      table.insert(ordered_keys, k)
    end
    table.sort(ordered_keys)
    local i = 1
    -- how many bytes have been sent on this client
    local clients_state = { }
    if nr_incoming > 0 then
      while true do
        thread_print("Waiting for number "..i.." incoming connections")
        server:clients(i, function(client)
          -- skip the clients that we have connected to
          if (client:tag() and tonumber(client:tag()) < i) then
            return
          end
          client:tag(tostring(i))
          -- current status: first tensor, 0 element sent
          table.insert(clients_state, {current_tensor=1, elem_sent=0})
          i = i + 1
          if (client:recv() ~= 'ver') then
            error('client protocol mismatch!')
          end
          thread_print('server received ver from client')
          -- server sends version message to client
          client:send('0.1')
        end)
        if debug then
          -- posix.sleep(1)
        end
        thread_print(i.." incoming connections")
        if i > nr_incoming then
          break
        end
      end
    end
    -- generate ordered keys for the table of parameters
    send_partial_tensor = function(tensor, tensor_state, client)
      -- find the start and end pos
      local ret = false
      local n_elem = tensor:nElement()
      local start_pos = tensor_state.elem_sent
      if start_pos > n_elem then
        return true
      end
      local end_pos = tensor_state.elem_sent + chunk_size - 1
      if end_pos >= n_elem then
        end_pos = n_elem
        ret = true
      end
      thread_print(string.format("client %s send %d from %d to %d", client:tag(), tensor_state.current_tensor, start_pos, end_pos))
      client:send(tensor[{{start_pos, end_pos}}])
      -- wait for sending complete! We must do this on old kernel
      -- posix.nanosleep(0,chunk_size / (1000000000 / 8) * 1000000000)
      -- posix.nanosleep(0,chunk_size * 8)
      tensor_state.elem_sent = end_pos + 1
      return ret
    end
    -- wait for all threads be ready
    -- barrier
    thread_print("Waiting for training starts!")
    server_sync_lock:lock()
    thread_print("sleeping...")
    server_sync_progress[1] = server_sync_progress[1] + 1
    server_lock:unlock()
    server_sync_cond:wait(server_sync_lock)
    server_sync_lock:unlock()
    -- now, start the main service loop
    while true do
      -- wait for the next trigger
      thread_print("All clients connected. Waiting for requests!")
      -- copy model parameters from gpu buffer if necessary
      if copy_from_gpu then
        self_parameters["_1"]:copy(gpu_buffer)
      end
      local finished_clients = 0
      -- unchoke all clients
      server:unchokeAll()
      thread_print("All clients unchoked")
      -- polling loop
      while true do
        local need_wait = true
        thread_print("waiting for connection...")
        server:clients(function(client)
          thread_print("checking client "..client:tag())
          local k = tonumber(client:tag())
          local state = clients_state[k]
          if state.current_tensor == 1 and state.elem_sent == 0 then
            -- have not received getblock yet
            if (client:recvAsync() == 'getblock') then
              -- received getblock request!
              -- prepare to send
              state.elem_sent = 1
              thread_print(string.format("client %s received getblock", client:tag()))
              need_wait = false
            else
              thread_print("not received")
            end
          elseif state.current_tensor > 0 then
            -- we are sending data, don't wait
            need_wait = false
            -- get last progress
            local key = ordered_keys[state.current_tensor]
            local tensor = self_parameters[key]
            if send_partial_tensor(tensor, state, client) then
              -- if returned true, this tensor has been sent
              local next_key = ordered_keys[state.current_tensor + 1]
              if next_key then
                -- move to the next tensor
                state.current_tensor = state.current_tensor + 1
                state.elem_sent = 1
              else
                -- no tensors left, we are done
                state.current_tensor = -1
                finished_clients = finished_clients + 1
                -- choke this client, don't receive data from it for this iteration
                client:choke(1)
              end
            end
          else
            thread_print(string.format("already done, finished %d, all %d", finished_clients, #clients_state))
          end
        end)
        if finished_clients == #clients_state then
          break
        end
        -- posix.nanosleep(0, debug and 300000000 or 100000)
        -- need to wait only if we are not sending data to any clients
        if need_wait then
          thread_print("wait...")
          -- posix.nanosleep(0, debug and 300000000 or 100000)
          server:waitAny()
          thread_print("wait done...")
        end
      end
      -- clear the clients_state for next iteration
      for k, v in pairs(clients_state) do
        v.current_tensor = 1
        v.elem_sent = 0
      end
      thread_print("tensors sent to all clients!")
      if early_average then
        -- do the average in server thread!
        -- wait for client done
        client_lock:lock()
        client_lock:unlock()
        for key, tensor in pairs(self_parameters) do
          if n_clients_weights > 0 then
            -- print("average: ", torch.sum(t_recv[key][1]))
            for j = 2,n_clients_weights do
              t_recv[key][1]:add(clients_weights[j], t_recv[key][j])
            end
            torch.add(tensor, t_recv[key][1], self_weight, tensor)
          else
            -- keep_model must be false
            tensor:mul(self_weight)
          end
        end
        -- also copy the new model parameter to GPU memory
        if copy_from_gpu then
          gpu_input_weights:copy(self_parameters["_1"])
        end
      end
      -- barrier
      server_sync_lock:lock()
      server_sync_progress[1] = server_sync_progress[1] + 1
      server_lock:unlock()
      thread_print("sleeping..."..client_sync_progress[1])
      server_sync_cond:wait(server_sync_lock)
      server_sync_lock:unlock()
      if (exit_flag[1] == 1) then
        thread_print("server exiting")
        break
      end
    end
    server:close()
  end

  -- define client thread
  local function make_client_thread(peer)
    return function(tid, client_sync_lock_id, client_sync_cond_id, client_lock_id)
      local ipc = require 'libipc'
      local threads = require 'threads'
      local posix = require 'posix'
      local client_sync_lock = threads.Mutex(client_sync_lock_id)
      local client_sync_cond = threads.Condition(client_sync_cond_id)
      local client_lock = threads.Mutex(client_lock_id)
      posix.sleep(5)
      thread_print = function(str)
        if debug then
          local colors = require 'ansicolors'
          local sec, nano
          sec, nano = posix.clock_gettime('CLOCK_MONOTONIC')
          print(colors(string.format('%%{%d}[%10d.%9d][CLIENT %d/%d %d] %s', tid+1, sec, nano, tid, total_clients, __threadid, str)))
        end
      end
      thread_print(string.format("Thread %d mutex ID is %d", tid, client_sync_lock_id))
      local client_ip = peer["host"]
      local client_port = peer["port"]
      thread_print(string.format("connecting to %s:%d", client_ip, client_port))
      local client = ipc.client(client_ip, client_port)
      thread_print(string.format("connected to %s:%d!", client_ip, client_port))
      client:send('ver')
      thread_print(string.format("version message sent"))
      if (client:recv() ~= '0.1') then
        error('server protocol mismatch')
      end
      thread_print('got 0.1 from server')
      -- generate ordered keys for the table of parameters
      local ordered_keys = {}
      for k in pairs(t_recv) do
        table.insert(ordered_keys, k)
      end
      table.sort(ordered_keys)
      -- wait for all threads be ready
      -- barrier
      thread_print("Waiting for training starts!")
      client_sync_lock:lock()
      client_sync_progress[1] = client_sync_progress[1] + 1
      if client_sync_progress[1] == total_clients then
        -- all client threads finished
        client_lock:unlock()
      end
      thread_print("sleeping..."..client_sync_progress[1])
      client_sync_cond:wait(client_sync_lock)
      client_sync_lock:unlock()
      while true do
        thread_print('requesting tensor!')
        client:send('getblock')
        thread_print(string.format('receiving tensor for thread %d', tid))
        if debug then
          -- posix.sleep(1)
        end
        -- receive tensor blocks, each one is 16 KB
        local current_tensor = 1
        local ind_recv = 1
        while true do
          local key = ordered_keys[current_tensor]
          if key then
            local tensor = t_recv[key][tid]
            local ind_end = ind_recv + chunk_size - 1;
            local new_ind_recv = ind_end + 1
            if ind_end >= tensor:nElement() then
              -- this tensor is done, start the next one
              ind_end = tensor:nElement()
              current_tensor = current_tensor + 1
              new_ind_recv = 1
            end
            thread_print(string.format("receiving %d size %d from %d to %d", current_tensor, tensor:nElement(), ind_recv, ind_end))
            client:recv(t_recv[key][tid][{{ind_recv, ind_end}}])
            ind_recv = new_ind_recv
          else
            break
          end
        end
        if tid == 1 then
          -- do the multiplication here, instead of during averaging
          for i, key in pairs(ordered_keys) do
            t_recv[key][1]:mul(clients_weights[1])
          end
        end
        thread_print('received tensor')
        -- print("receiving sum: ", torch.sum(t_recv["_1"][tid]))
        -- barrier
        client_sync_lock:lock()
        client_sync_progress[1] = client_sync_progress[1] + 1
        if client_sync_progress[1] == total_clients then
          -- all client threads finished
          client_lock:unlock()
        end
        thread_print("sleeping..."..client_sync_progress[1])
        client_sync_cond:wait(client_sync_lock)
        client_sync_lock:unlock()
        if (exit_flag[1] == 1) then
          thread_print("client exiting")
          break
        end
      end
      client:close()
    end
  end

  -- create additional threads for clients and servers
  local function CreateClientServerThreads()

    -- make client threads, and save corresponding weights
    for i, pw in pairs(peer_weights) do
      if pw ~= 0 then
        table.insert(clients_weights, pw)
        table.insert(clients, make_client_thread(nodes[i]))
      end
    end
    n_clients_weights = #clients_weights
    
    local prev_type = torch.getdefaulttensortype()
    -- create temporary tensors for receiving
    for key, tensor in pairs(self_parameters) do
      local new_size = torch.totable(tensor:size())
      table.insert(new_size, 1, #clients)
      print("creating temp tensor with size ", new_size)
      -- t_recv:resize(table.unpack(new_size))
      torch.setdefaulttensortype(tensor:type())
      t_recv[key] = torch.Tensor(table.unpack(new_size))
      -- fill it with NaN for fault detection
      t_recv[key]:fill(0/0)
    end
    torch.setdefaulttensortype(prev_type)

    -- create threads
    print(clients)
    total_clients = #clients
    print("start creating servers with transmission chunk size "..chunk_size)
    -- now start our server and client thread
    pool = threads.Threads(1 + #clients, function(threadid)
                                           require 'torch'
                                           if cuda then
                                             require 'cutorch'
                                           end
                                           torch.setnumthreads(1)
                                         end)
    print("start creating clients")
    -- server is using the model parameter, hold the lock
    server_lock:lock()
    -- client is using the model parameter, hold the lock
    client_lock:lock()
    pool:addjob(server_thread, function() end, server_sync_lock_id, server_sync_cond_id, server_lock_id, client_lock_id)
    for i = 1,#clients do
      pool:addjob(clients[i], function() end, i, client_sync_lock_id, client_sync_cond_id, client_lock_id)
    end
    thread_print = function(str)
      if debug then
        local colors = require 'ansicolors'
        local sec, nano
        sec, nano = posix.clock_gettime('CLOCK_MONOTONIC')
        print(colors(string.format('%%{%d}[%10d.%9d][SERVER 0] %s', #clients + 2, sec, nano, str)))
      end
    end

  end

  local function Init()
    ExpandParameters(model_parameters, '')
    print(self_parameters)
    GetSelfIP()
    GetSelfWeight()
    CreateClientServerThreads()
    return self_rank
  end

  local function StartCommunication()
    -- pool:synchronize()
    -- wait for all peers are ready
    -- wait for all client threads
    while true do
      client_sync_lock:lock()
      local nr_connected = client_sync_progress[1]
      print("Waiting for client threads, "..nr_connected.." of "..total_clients)
      if (nr_connected == total_clients) then
        thread_print("All clients connected, progress="..client_sync_progress[1])
        client_sync_progress[1] = 0
        client_lock:lock()
        client_sync_cond:broadcast()
        client_sync_lock:unlock()
        break
      else
        client_sync_lock:unlock()
      end
      posix.sleep(2)
    end
    print("Waiting for server thread...")
    -- wait until server release the parameter lock
    while true do
      server_sync_lock:lock()
      if (server_sync_progress[1] == 1) then
        thread_print("Server started!")
        server_sync_progress[1] = 0
        server_lock:lock()
        server_sync_cond:broadcast()
        server_sync_lock:unlock()
        break
      else
        server_sync_lock:unlock()
      end
      posix.sleep(1)
    end
    print("Starting Training!")
  end

  local function CheckIfClientSyncDone()
    client_sync_lock:lock()
    if (client_sync_progress[1] == total_clients) then
      thread_print("Clients synced, progress="..client_sync_progress[1])
      client_sync_lock:unlock()
      return true
    end
    client_sync_lock:unlock()
    return false
  end

  local function CheckIfServerSyncDone()
    server_sync_lock:lock()
    -- currently we only have 1 server
    if (server_sync_progress[1] == 1) then
      thread_print("Server synced, progress="..server_sync_progress[1])
      server_sync_lock:unlock()
      return true
    end
    server_sync_lock:unlock()
    return false
  end

  local function CheckIfSyncDone()
    ret = CheckIfClientSyncDone() and CheckIfServerSyncDone()
    -- pool:synchronize()
    return ret
  end

  local function WaitForClientSyncDone()
    -- try to gain the client lock
    client_lock:lock()
    assert(CheckIfClientSyncDone(), "CheckIfClientSyncDone() should return true")
    client_lock:unlock()
  end

  local function WaitForServerSyncDone()
    -- try to gain the server lock
    server_lock:lock()
    assert(CheckIfServerSyncDone(), "CheckIfServerSyncDone() should return true")
    server_lock:unlock()
  end

  -- if keep_model is set to true, we don't put results to self_weight, as the server thread
  -- may be still using them (this only works when there are at least 1 client!)
  local function AverageParameters(keep_model)
    if early_average then
      error("AverageParameters should not be called when early_average is set to true")
    end
    local ret = { }
    -- we have sent tensors to all peers and got all tensors from peers, now do an average
    for key, tensor in pairs(self_parameters) do
      --[[
      if tensor:isContiguous() and not string.find(torch.typename(tensor), 'Cuda') then
      -- if false then 
        local self_data = torch.data(tensor)
        local clients_data = {}
        for j = 1,n_clients_weights do
          table.insert(clients_data, torch.data(t_recv[key][j]))
        end
        -- access cdata
        for i = 0, tensor:nElement()-1 do
          local s = self_data[i] * self_weight
          for j = 1,n_clients_weights do
            s = s + clients_data[j][i] * clients_weights[j]
          end
          self_data[i] = s
        end
      else
      --]]
        local output
        if keep_model then
          -- set the output to t_recv[key][1] if it is available
          output = t_recv[key] and t_recv[key][1] or tensor
        else
          output = tensor
        end
        if n_clients_weights > 0 then
          -- print("average: ", torch.sum(t_recv[key][1]))
          for j = 2,n_clients_weights do
            t_recv[key][1]:add(clients_weights[j], t_recv[key][j])
          end
          torch.add(output, t_recv[key][1], self_weight, tensor)
        else
          -- keep_model must be false
          output:mul(tensor, self_weight)
        end
        ret[key] = output
      --end
    end
    return ret
  end

  local function StartNextIter()
    -- start next iteration
    thread_print("staring next iteration...")
    client_lock:lock()
    client_sync_lock:lock()
    client_sync_progress[1] = 0
    client_sync_cond:broadcast()
    client_sync_lock:unlock()
    server_lock:lock()
    server_sync_lock:lock()
    server_sync_progress[1] = 0
    server_sync_cond:broadcast()
    server_sync_lock:unlock()
    thread_print("go!")
  end

  local function SetExitFlag()
    exit_flag[1] = 1
  end

  local function Terminate()
    pool:synchronize()
    print("exiting...")
    pool:terminate()
    client_sync_lock:free()
    client_sync_cond:free()
    client_lock:free()
    server_sync_lock:free()
    server_sync_cond:free()
    server_lock:free()
  end

  return {
    Init = Init,
    StartCommunication = StartCommunication,
    CheckIfSyncDone = CheckIfSyncDone,
    CheckIfClientSyncDone = CheckIfClientSyncDone,
    CheckIfServerSyncDone = CheckIfServerSyncDone,
    WaitForClientSyncDone = WaitForClientSyncDone,
    WaitForServerSyncDone = WaitForServerSyncDone,
    AverageParameters = AverageParameters,
    StartNextIter = StartNextIter,
    SetExitFlag = SetExitFlag,
    Terminate = Terminate
  }
end

local function LoadConfigFromFile(nodes_file, weights_file)
  -- read all nodes
  io.input(nodes_file)
  local nodes = { }
  -- read the lines in table 'lines'
  for line in io.lines() do
    line = stringx.strip(line)
    if (line ~= '') then
      hostport = string.split(line,':')
      table.insert(nodes, {host=hostport[1], port=hostport[2], self=false})
    end
  end
  print(nodes)

  -- read all weights
  io.input(weights_file)
  local weights = { }
  for line in io.lines() do
    line = stringx.strip(line)
    if (line ~= '') then
      local line_weights = string.split(line,' ')
      for i,v in pairs(line_weights) do
        line_weights[i] = tonumber(v)
      end
      table.insert(weights, line_weights)
    end
  end
  print(weights)
  return nodes, weights
end

return {
  Trainer = DecentralizedSGD,
  LoadConfigFromFile = LoadConfigFromFile
}

