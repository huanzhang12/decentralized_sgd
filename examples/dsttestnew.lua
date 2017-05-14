local DecentralizedSGD = require 'dstsgd'
local posix = require 'posix'

local opt = lapp [[
Decentralized SGD testing

   --nodesFile         (default 'nodes.txt')    A text file with all host names and port number
   --weightsFile       (default 'weights.txt')  A text file with weights for parameters from different machines
   --nodeID            (default 0)              Which node is this machine? Set 0 for auto
   --loops             (default 20)             How many seconds to test
   --chunkSize         (default 16384)          Transfer chunk size
]]

require 'cutorch'

-- The shared tensor, just for testing
local t = {tensor1 = torch.FloatTensor(3,16384):fill(opt.nodeID):cuda(),
           tensor2 = torch.DoubleTensor(256,128):fill(torch.uniform()),
           tensor3 = torch.FloatTensor(256,512):fill(torch.uniform()):cuda()
          }

-- load nodes and weights from a file
nodes, weights = DecentralizedSGD.LoadConfigFromFile(opt.nodesFile, opt.weightsFile)

-- create decentralized trainer object
dstsgd = DecentralizedSGD.Trainer(nodes, weights, opt.nodeID, t, true, opt.chunkSize)

print("Start init")
dstsgd.Init()
print("Init done.")
-- create model, etc, while waiting all nodes to connect
dstsgd.StartCommunication()
print("Ready to train!")

for i = 1,opt.loops do
  print("Iteration ", i)
  dstsgd.WaitForClientSyncDone()
  print("Averaging...")
  local tmp_tensors = dstsgd.AverageParameters(true)
  print(t.tensor1[1][1], t.tensor1[3][16384])
  print(t.tensor2[1][1], t.tensor2[256][128])
  print(t.tensor3[1][1], t.tensor3[256][512])
  dstsgd.WaitForServerSyncDone()
  t.tensor1:copy(tmp_tensors._tensor1)
  t.tensor2:copy(tmp_tensors._tensor2)
  t.tensor3:copy(tmp_tensors._tensor3)
  print(t.tensor1[1][1], t.tensor1[3][16384])
  print(t.tensor2[1][1], t.tensor2[256][128])
  print(t.tensor3[1][1], t.tensor3[256][512])
  if i == opt.loops then
    dstsgd.SetExitFlag()
  end
  dstsgd.StartNextIter()
end

dstsgd.Terminate()

