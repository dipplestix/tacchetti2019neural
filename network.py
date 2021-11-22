import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AvgPool2d
from torch.optim import Adam, SGD


class AuctionNet(Module):   
    def __init__(self, players, items):
        super(AuctionNet, self).__init__()
        self.players = players
        self.items = items
        
        self.cnn_layers = Sequential(
            Conv2d(items*2 + 1, 64, kernel_size=[1, 1]),
            ReLU(inplace=True),
            
            Conv2d(64, 64, kernel_size=[1, items]),            
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(64, 64), 
            ReLU(inplace=True), 
            Linear(64, 64), 
            ReLU(inplace=True)
        )
                
        # self.decoder = Sequential(Linear(64, 1), ReLU(inplace=True))
        # Paper says this should be ReLU, but that breaks training on a lot of cases
        self.decoder = Sequential(Linear(64, 1))

        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.reshape(x, (-1, self.players-1, 64))
        y = self.linear_layers(x)
        x = torch.sum(y, 1)
        x =self.decoder(x)
        return x


class AucLoss(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, t, results, lam_b=100, lam_r=100):
        sum_t = torch.sum(t)
        # print(f'sum_t: {sum_t}')
        budget_balanced = torch.square(torch.minimum(torch.sum(t), torch.zeros([1])))
        # print(f'budget_balanced: {budget_balanced}')
        rationality = torch.sum(torch.square(torch.minimum(results - t, torch.zeros(t.shape))))
        # print(f'rationality: {rationality}')
        loss = sum_t + lam_b*budget_balanced + lam_r*rationality
        return loss

    
def process(vals):
    # vals = torch.tensor(vs)

    players = vals.shape[0]
    items = vals.shape[1]
    
    index = torch.topk(vals.flatten(), items).indices
    winners = torch.zeros(vals.flatten().shape)
    winners[index] = 1
    winners = winners.reshape(vals.shape)
    results = winners*vals
    
    other_results = []
    cf_vals = []
    
    for i in range(players):
        player_index = [p for p in range(players) if p != i]
        
        loop_results = results[player_index]
        other_results.append(torch.sum(loop_results))
        
        loop_vals = vals[player_index]
        cf_index = torch.topk(loop_vals.flatten(), items).indices
        cf_winners = torch.zeros(loop_vals.flatten().shape)
        cf_winners[cf_index] = 1
        cf_winners = cf_winners.reshape(loop_vals.shape)
        cf_results = cf_winners*loop_vals
        cf_vals.append(torch.sum(cf_results))
        
        loop_vals = torch.unsqueeze(loop_vals, 0)
        for k in range(items):
            cf_index = torch.topk(loop_vals[0].flatten(), k+1).indices
            l = torch.zeros(loop_vals[0].flatten().shape)
            l[cf_index] = 1
            l = l.reshape(loop_vals[0].shape)
            l = torch.unsqueeze(l, 0)
            loop_vals = torch.cat((loop_vals, l))
            l = loop_vals[0]*l
            loop_vals = torch.cat((loop_vals, l))
        
        if i == 0:
            loop_vals = torch.unsqueeze(loop_vals, 0)
            tensors = loop_vals
        else:
            loop_vals = torch.unsqueeze(loop_vals, 0)
            tensors = torch.cat((tensors, loop_vals))

    return torch.stack(other_results), torch.stack(cf_vals), torch.sum(results, 1), tensors
