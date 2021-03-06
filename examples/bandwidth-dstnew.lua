local DecentralizedSGD = require 'dstsgd'
local posix = require 'posix'

local opt = lapp [[
Decentralized SGD testing

   --nodesFile         (default 'nodes.txt')    A text file with all host names and port number
   --weightsFile       (default 'weights.txt')  A text file with weights for parameters from different machines
   --nodeID            (default 0)              Which node is this machine? Set 0 for auto
   --gpu               (default 0)              Use CUDA tensor
   --loops             (default 20)             How many seconds to test
   --chunkSize         (default 16384)          Transfer chunk size
]]


-- The shared tensor, just for testing
local t = { }
local use_gpu = false
if opt.gpu == 1 then
  require 'cutorch'
  t.tensor1 = torch.FloatTensor(270410):fill(opt.nodeID):cuda();
  use_gpu = true
else
  t.tensor1 = torch.FloatTensor(270410):fill(opt.nodeID);
end

torch.setnumthreads(1)

-- load nodes and weights from a file
nodes, weights = DecentralizedSGD.LoadConfigFromFile(opt.nodesFile, opt.weightsFile)

-- create decentralized trainer object
dstsgd = DecentralizedSGD.Trainer(nodes, weights, opt.nodeID, t, use_gpu, opt.chunkSize)

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
  dstsgd.WaitForClientSyncDone()
  local tmp_tensor = dstsgd.AverageParameters(true)
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
  dstsgd.WaitForServerSyncDone()
  dstsgd.StartNextIter()
end

dstsgd.Terminate()

