local opt = lapp [[
Train a CNN classifier on MNIST using DecentralizedSGD.

   --nodesFile         (default 'nodes.txt')    A text file with all host names and port number
   --weightsFile       (default 'weights.txt')  A text file with weights for parameters from different machines
   --nodeID            (default 0)              Which node is this machine? Set 0 for auto
]]

-- Requires
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'
local DecentralizedSGD = require 'dstsgd'


-- load nodes and weights from a file
nodes, weights = DecentralizedSGD.LoadConfigFromFile(opt.nodesFile, opt.weightsFile)


torch.setnumthreads(1)

-- Load the MNIST dataset
local trainingDataset = Dataset('http://d3jod65ytittfm.cloudfront.net/dataset/mnist/train.t7', {
   partition = opt.nodeID,
   partitions = #nodes,
})
-- leave some data for validation
-- all nodes have the same validation set
validationDataset = torch.load('train.t7')

local getTrainingBatch, numTrainingBatches = trainingDataset.sampledBatcher({
   samplerKind = 'permutation',
   batchSize = 1,
   inputDims = { 1024 },
   verbose = true,
   processor = function(res, processorOpt, input)
      input:copy(res:view(1, 1024))
      return true
   end,
})

-- Load in MNIST
local classes = {'0','1','2','3','4','5','6','7','8','9'}
local confusionMatrix = optim.ConfusionMatrix(classes)

-- Ensure params are equal on all nodes
torch.manualSeed(0)

-- What model to train:
local predict,f,params

-- for CNNs, we rely on efficient nn-provided primitives:
local reshape = grad.nn.Reshape(1,32,32)

local conv1, acts1, pool1, conv2, acts2, pool2, flatten, linear
local params = {}
conv1, params.conv1 = grad.nn.SpatialConvolutionMM(1, 16, 5, 5)
acts1 = grad.nn.Tanh()
pool1 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

conv2, params.conv2 = grad.nn.SpatialConvolutionMM(16, 16, 5, 5)
acts2 = grad.nn.Tanh()
pool2, params.pool2 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

flatten = grad.nn.Reshape(16*5*5)
linear,params.linear = grad.nn.Linear(16*5*5, 10)

-- Cast the parameters
params = grad.util.cast(params, 'float')

-- create decentralized trainer object
dstsgd = DecentralizedSGD.Trainer(nodes, weights, opt.nodeID, params)

print("Start init")
dstsgd.Init()
print("Init done.")

-- Define our network
function predict(params, input, target)
   local h1 = pool1(acts1(conv1(params.conv1, reshape(input))))
   local h2 = pool2(acts2(conv2(params.conv2, h1)))
   local h3 = linear(params.linear, flatten(h2))
   local out = util.logSoftMax(h3)
   return out
end

-- Define our loss function
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, {
   optimize = true,              -- Generate fast code
   stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
})

local function table_parameters_func(elem1, elem2, op)
   if (type(elem1) == 'table') then
      local ordered_keys = {}
      for k in pairs(elem1) do
        table.insert(ordered_keys, k)
      end
      table.sort(ordered_keys)
      for i = 1, #ordered_keys do
         key = ordered_keys[i]
         elem = elem1[key]
         if elem2[key] == nil then
            if type(elem) == 'table' then
               elem2[key] = { }
            else
               elem2[key] = torch.FloatTensor()
            end
         end
         table_parameters_func(elem, elem2[key], op)
      end
   else
      op(elem1, elem2) 
   end
end

-- initialization of parameters
torch.manualSeed(0)
table_parameters_func(params, {}, function(elem1, elem2) torch.randn(elem1, elem1:size()) elem1:mul(0.01) end)

dstsgd.StartCommunication()
print("Ready to train!")

local grads = { }
-- Train a neural network
for epoch = 1,100 do
   print('Training Epoch #'..epoch)
   -- Evaluate on the test set for each thread and see if they match
   confusionMatrix:zero()
   for i = 1, 1000 do
      -- local pred = predict(allReduceEA.center, validationDataset.x[{i}])
      local pred = predict(params, validationDataset.x[{i}])
      local expected = util.oneHot(validationDataset.y[{{i}}], 10)
      -- thread_print(pred, expected)
      confusionMatrix:add(pred[1], expected[1])
   end
   print(confusionMatrix)
   local loss, prediction
   local j = 1
   for i = 1,numTrainingBatches() do
      -- Next sample:
      local batch = getTrainingBatch()
      local x = batch.input
      local y = util.oneHot(batch.target, 10)

      -- Grads:
      local b_grads = df(params,x,y)
      if j == 1 then
         table_parameters_func(b_grads, grads, function(elem1, elem2) elem2:resize(elem1:size()) elem2:copy(elem1) end)
      else
         -- accumulate the gradients
         table_parameters_func(grads, b_grads, function(elem1, elem2) elem1:add(elem2) end)
      end

      if dstsgd.CheckIfSyncDone() then
         -- parameter transmission done. Start updating parameter of this node
         dstsgd.AverageParameters()
      -- if j > 10 then
         -- print(j)
         table_parameters_func(grads, {}, function(elem1, elem2) elem1:mul(1.0/j) end)
         -- Update weights and biases
         for iparam=1,2 do
            params.conv1[iparam] = params.conv1[iparam] - grads.conv1[iparam] * 0.01
            params.conv2[iparam] = params.conv2[iparam] - grads.conv2[iparam] * 0.01
            params.linear[iparam] = params.linear[iparam] - grads.linear[iparam] * 0.01
         end
         j = 0
      end
      j = j + 1
   end
end

