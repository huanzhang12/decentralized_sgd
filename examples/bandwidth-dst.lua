local DecentralizedSGD = require 'dstsgd'
local posix = require 'posix'

local opt = lapp [[
Decentralized SGD testing

   --nodesFile         (default 'nodes.txt')    A text file with all host names and port number
   --weightsFile       (default 'weights.txt')  A text file with weights for parameters from different machines
   --nodeID            (default 0)              Which node is this machine? Set 0 for auto
   --gpu               (default 0)              Use CUDA tensor
   --loops             (default 20)             How many seconds to test
]]


-- The shared tensor, just for testing
local t = { }
if opt.gpu == 1 then
  require 'cutorch'
  t.tensor1 = torch.FloatTensor(1024,1024):fill(opt.nodeID):cuda();
else
  t.tensor1 = torch.FloatTensor(1024,1024):fill(opt.nodeID);
end


-- load nodes and weights from a file
nodes, weights = DecentralizedSGD.LoadConfigFromFile(opt.nodesFile, opt.weightsFile)

-- create decentralized trainer object
dstsgd = DecentralizedSGD.Trainer(nodes, weights, opt.nodeID, t)

print("Start init")
dstsgd.Init()
print("Init done.")
-- create model, etc, while waiting all nodes to connect
dstsgd.StartCommunication()
print("Ready to train!")

timer = torch.Timer()
local i = 0
local j = 0
while true do
  i = i + 1
  local retry = 0
  while true do
    -- compute gradients, etc
    posix.nanosleep(0,100000)
    retry = retry + 1
    -- check the atomic counter to see if we have finished communication
    if dstsgd.CheckIfSyncDone() or (retry > 50000) then
      break
    end
  end
  dstsgd.AverageParameters()
  if j == -1 then
    break
  end
  if timer:time().real > 1.0 then
    timer:stop()
    print(string.format("%d loops in %.3f seconds", i, timer:time().real))
    i = 0
    j = j + 1
    timer:reset()
    timer:resume()
  end
  if j >= opt.loops then
    dstsgd.SetExitFlag()
    j = - 1
  end
end

dstsgd.Terminate()

