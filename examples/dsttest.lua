local DecentralizedSGD = require 'dstsgd'
local posix = require 'posix'

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

-- read the weights

-- now load the weight averaging matrix, and determine which machine we want to connect to
io.input(opt.weightsFile)
local weights = { }
-- read the lines in table 'lines'
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

-- The shared tensor, just for testing
local t = torch.FloatTensor(1024,1024):fill(opt.nodeID)

-- create decentralized trainer object
dstsgd = DecentralizedSGD(nodes, weights, opt.nodeID, t)

print("Start init")
dstsgd.Init()
print("Init done.")
-- create model, etc, while waiting all nodes to connect
dstsgd.StartCommunication()
print("Ready to train!")

for i = 1,10 do
  print("Iteration "..i)
  while true do
    -- compute gradients, etc
    posix.sleep(1)
    print("Computing gradients...")
    -- check the atomic counter to see if we have finished communication
    if dstsgd.CheckIfSyncDone() then
      break
    end
  end
  if i == 10 then
    dstsgd.SetExitFlag()
  end
  print("Averaging...")
  dstsgd.AverageParameters()
  print(t[1][1])
end

dstsgd.Terminate()

