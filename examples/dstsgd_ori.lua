local ipc = require 'libipc'
local threads = require 'threads'
local tds = require 'tds'
-- share upvalue tensors between threads
threads.serialization('threads.sharedserialize')
local colors = require 'ansicolors'



local opt = lapp [[
Decentralized SGD testing

   --nodesFile         (default 'nodes.txt')    A text file with all host names and port number
   --weightsFile       (default 'weights.txt')  A text file with weights for parameters from different machines
   --nodeID            (default 0)              Which node is this machine? Set 0 for auto
]]

-- read all nodes
io.input(opt.nodesFile)
local nodes = {}
-- read the lines in table 'lines'
for line in io.lines() do
  line = stringx.strip(line)
  if (line ~= '') then
    hostport = string.split(line,':')
    table.insert(nodes, {host=hostport[1], port=hostport[2], self=false})
  end
end
print(nodes)

-- find IP of this node
local self_ip, self_port, self_rank, self_weight
if opt.nodeID == 0 then
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
    print('Cannot found self IP in nodes list. Try to set nodeID.')
    return
  end
else
  self_rank = opt.nodeID
  self_ip = nodes[opt.nodeID]["host"]
  self_port = nodes[opt.nodeID]["port"]
  nodes[self_rank]["self"] = true
end
print(string.format("This machine is %s:%s, rank %d", self_ip, self_port, self_rank))

-- print each thread message in different color
thread_print = function(str)
  print(colors('%{'..self_rank..'}'..tostring(str)))
end

-- now load the weight averaging matrix, and determine which machine we want to connect to
io.input(opt.weightsFile)
local peer_weights
local nr_incoming = 0
local nr_lines = 0
-- read the lines in table 'lines'
for line in io.lines() do
  line = stringx.strip(line)
  if (line ~= '') then
    nr_lines = nr_lines + 1
    local weights = string.split(line,' ')
    if tonumber(weights[self_rank]) ~= 0 then
      nr_incoming = nr_incoming + 1
    end
    if (nr_lines == self_rank) then
      peer_weights = string.split(line,' ')
      -- remove ourself from clients
      nr_incoming = nr_incoming - 1
      for i,v in pairs(peer_weights) do
        peer_weights[i] = tonumber(v)
      end
      self_weight = peer_weights[self_rank]
      peer_weights[self_rank] = 0.0
    end
  end
end
print(self_weight)
print(peer_weights)
print(nr_incoming)

-- The shared tensor, just for testing
-- The server thread share the tensor with main training thread!
local t = torch.FloatTensor(1024,1024):fill(self_rank)
local t_recv = torch.FloatTensor(3, 1024,1024):fill(1)
-- shared flag for exiting worker threads
local exit_flag = torch.IntTensor(1):fill(0)
-- mutex and conditional variable for iteration synchronization between main thread and communication threads
local sync_lock = threads.Mutex()
local sync_lock_id = sync_lock:id()
local sync_cond = threads.Condition()
local sync_cond_id = sync_cond:id()
-- the counter will +1 after send/get one weight
local sync_progress = tds.AtomicCounter()

server_thread = function(sync_lock_id, sync_cond_id) 
  local ipc = require 'libipc'
  local server = ipc.server(self_ip, self_port)
  local colors = require 'ansicolors'
  local threads = require 'threads'
  local sync_lock = threads.Mutex(sync_lock_id)
  local sync_cond = threads.Condition(sync_cond_id)
  thread_print = function(str)
    print(colors('%{1}[SERVER 0]'..tostring(str)))
  end
  thread_print(string.format("mutex ID is %d", sync_lock_id))
  -- on initialization, wait for num_clients clients
  thread_print(string.format('Server at %s:%d waiting...', self_ip, self_port))
  -- initialize connection
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
    os.execute("sleep 1")
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
        client:send(t)
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


function make_client_thread(peer)
  return function(tid, sync_lock_id, sync_cond_id)
    os.execute("sleep 10")
    local ipc = require 'libipc'
    local colors = require 'ansicolors'
    local threads = require 'threads'
    local sync_lock = threads.Mutex(sync_lock_id)
    local sync_cond = threads.Condition(sync_cond_id)
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
      os.execute("sleep 1")
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

-- Now figure out how many outcoming connections we need to make
local nr_peers = 0
local clients = { }
local clients_weights = { }
for i, pw in pairs(peer_weights) do
  if pw ~= 0 then
    nr_peers = nr_peers + 1
    table.insert(clients, make_client_thread(nodes[i]))
    table.insert(clients_weights, pw)
  end
end

print(clients)

print("start creating servers")
-- now start our server and client thread
local pool = threads.Threads(1 + #clients)
print("start creating clients")
pool:addjob(server_thread, function() end, sync_lock_id, sync_cond_id)
for i = 1,#clients do
  pool:addjob(clients[i], function() end, i, sync_lock_id, sync_cond_id)
  -- pool:addjob(testfunc)
end

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
  os.execute("sleep 5")
end


for i = 1,10 do
  print("Iteration "..i)
  while true do
    -- compute gradients, etc
    os.execute("sleep 1")
    print("Computing gradients..."..sync_progress:get())
    -- check the atomic counter to see if we have finished communication
    sync_lock:lock()
    if (sync_progress:get() == total_threads) then
      sync_progress:set(0)
      print("Now averaging")
      break
    end
    sync_lock:unlock()
  end
  if i == 10 then
    exit_flag[1] = 1
  end
  -- we have sent tensors to all peers and got all tensors from peers, now do an average
  t:mul(self_weight)
  for j = 1,#clients_weights do
    t:add(clients_weights[j], t_recv[j])
  end
  print(t[1][1])
  -- start next iteration
  sync_cond:broadcast()
  sync_lock:unlock()
end
pool:synchronize()

print("exiting...")
pool:terminate()
sync_lock:free()
sync_cond:free()


