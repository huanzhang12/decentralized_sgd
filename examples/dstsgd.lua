local ipc = require 'libipc'
local threads = require 'threads'
local tds = require 'tds'
local colors = require 'ansicolors'
local posix = require 'posix'

-- nodes is a list of hostname and port numbers
-- node_weights is a matrix for parameter weights for different nodes
-- node_id is the ID of this node
-- self_parameters is a tensor of training parameters
local function DecentralizedSGD(nodes, node_weights, node_id, self_parameters)

  -- thread pool
  local pool
  -- network parameters of myself
  local self_ip, self_port, self_rank, self_weight
  -- weights for all connected peers
  local peer_weights
  -- number of incoming peers
  local nr_incoming
  -- shared flag for exiting worker threads
  local exit_flag = torch.IntTensor(1):fill(0)
  -- thread pool for clients and servers
  local pool
  -- threads for clients
  local clients = { }
  -- weights for connected clients
  local clients_weights = { }
  -- share upvalue tensors between threads
  threads.serialization('threads.sharedserialize')
  -- mutex and conditional variable for iteration synchronization between main thread and communication threads
  local sync_lock = threads.Mutex()
  local sync_lock_id = sync_lock:id()
  local sync_cond = threads.Condition()
  local sync_cond_id = sync_cond:id()
  -- the counter will +1 after send/get one weight
  local sync_progress = tds.AtomicCounter()
  -- create tensors for receiving tensors from peers
  torch.setdefaulttensortype(self_parameters:type())
  local t_recv = torch.Tensor()

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
        if hostport["host"] == ip then
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
  local server_thread = function(sync_lock_id, sync_cond_id) 
    local ipc = require 'libipc'
    local colors = require 'ansicolors'
    local threads = require 'threads'
    local posix = require 'posix'
    local sync_lock = threads.Mutex(sync_lock_id)
    local sync_cond = threads.Condition(sync_cond_id)
    thread_print = function(str)
      print(colors('%{1}[SERVER 0]'..tostring(str)))
    end
    thread_print(string.format("mutex ID is %d", sync_lock_id))
    -- initialize connection
    local server = ipc.server(self_ip, self_port)
    -- on initialization, wait for num_clients clients
    thread_print(string.format('Server at %s:%d waiting...', self_ip, self_port))
    local i = 1
    while true do
      thread_print("Waiting for "..i.." incoming connections")
      server:clients(i, function(client)
        -- skip the clients that we have connected to
        if (client:tag() and tonumber(client:tag()) < i) then
          return
        end
        client:tag(tostring(i))
        i = i + 1
        if (client:recv() ~= 'ver') then
          error('client protocol mismatch!')
        end
        thread_print('server received ver from client')
        -- server sends version message to client
        client:send('0.1')
      end)
      posix.sleep(1)
      thread_print(i.." incoming connections")
      if i > nr_incoming then
        break
      end
    end
    -- wait for all threads be ready
    -- barrier
    thread_print("Waiting for training starts!")
    sync_lock:lock()
    thread_print("sleeping...")
    sync_progress:inc()
    sync_cond:wait(sync_lock)
    sync_lock:unlock()
    -- now, start the main service loop
    while true do
      -- wait for the next trigger
      thread_print("All clients connected. Waiting for requests!")
      server:clients(function(client)
        if (client:recv() == 'getblock') then
          client:send(self_parameters)
        else
          error('unexpected command from client')
        end
      end)
      thread_print("tensors sent to all clients!")
      -- barrier
      sync_lock:lock()
      sync_progress:inc()
      sync_cond:wait(sync_lock)
      sync_lock:unlock()
      if (exit_flag[1] == 1) then
        thread_print("server exiting")
        break
      end
    end
    server:close()
  end

  -- define client thread
  local function make_client_thread(peer)
    return function(tid, sync_lock_id, sync_cond_id)
      local ipc = require 'libipc'
      local colors = require 'ansicolors'
      local threads = require 'threads'
      local posix = require 'posix'
      local sync_lock = threads.Mutex(sync_lock_id)
      local sync_cond = threads.Condition(sync_cond_id)
      posix.sleep(10)
      thread_print = function(str)
        print(colors('%{'..(tid+1)..'}[CLIENT '..tid..']'..tostring(str)))
      end
      thread_print(string.format("Thread %d mutex ID is %d", tid, sync_lock_id))
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
      -- wait for all threads be ready
      -- barrier
      thread_print("Waiting for training starts!")
      sync_lock:lock()
      thread_print("sleeping...")
      sync_progress:inc()
      sync_cond:wait(sync_lock)
      sync_lock:unlock()
      while true do
        thread_print('requesting tensor!')
        client:send('getblock')
        thread_print(string.format('receiving tensor for thread %d', __threadid))
        posix.sleep(1)
        client:recv(t_recv[tid])
        thread_print('received tensor')
        -- barrier
        sync_lock:lock()
        sync_progress:inc()
        sync_cond:wait(sync_lock)
        sync_lock:unlock()
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
        table.insert(clients, make_client_thread(nodes[i]))
        table.insert(clients_weights, pw)
      end
    end
    
    local new_size = torch.totable(self_parameters:size())
    table.insert(new_size,1,#clients)
    print("creating temp tensor with size ", new_size)
    t_recv:resize(table.unpack(new_size))

    -- create threads
    print(clients)
    print("start creating servers")
    -- now start our server and client thread
    pool = threads.Threads(1 + #clients)
    print("start creating clients")
    pool:addjob(server_thread, function() end, sync_lock_id, sync_cond_id)
    for i = 1,#clients do
      pool:addjob(clients[i], function() end, i, sync_lock_id, sync_cond_id)
    end

  end

  local function Init()
    GetSelfIP()
    GetSelfWeight()
    CreateClientServerThreads()
  end

  local function StartCommunication()
    -- wait for all peers are ready
    local total_threads = #clients + 1
    while true do
      sync_lock:lock()
      local nr_connected = sync_progress:get()
      print("Waiting for peers, "..nr_connected.." of "..total_threads)
      if (nr_connected == total_threads) then
        sync_progress:set(0)
        sync_cond:broadcast()
        sync_lock:unlock()
        break
      else
        sync_lock:unlock()
      end
      posix.sleep(5)
    end
  end

  local function CheckIfSyncDone()
    local total_threads = #clients + 1
    sync_lock:lock()
    if (sync_progress:get() == total_threads) then
      sync_progress:set(0)
      return true
      -- lock will be released in AverageParameters()
    end
    sync_lock:unlock()
    return false
  end

  local function AverageParameters()
    -- we have sent tensors to all peers and got all tensors from peers, now do an average
    self_parameters:mul(self_weight)
    for j = 1,#clients_weights do
      self_parameters:add(clients_weights[j], t_recv[j])
    end
    -- start next iteration
    sync_cond:broadcast()
    sync_lock:unlock()
  end

  local function SetExitFlag()
    exit_flag[1] = 1
  end

  local function Terminate()
    pool:synchronize()
    print("exiting...")
    pool:terminate()
    sync_lock:free()
    sync_cond:free()
  end

  return {
    Init = Init,
    StartCommunication = StartCommunication,
    CheckIfSyncDone = CheckIfSyncDone,
    AverageParameters = AverageParameters,
    SetExitFlag = SetExitFlag,
    Terminate = Terminate
  }
end

return DecentralizedSGD

