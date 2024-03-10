from torch import nn
class MLP(nn.Module):
    def __init__(self, d_in : int,hidden_layers : list[int], d_out: int,) -> None:
        super(MLP,self).__init__()
        self.linear = nn.ModuleList()
        layers = [d_in]+ hidden_layers + [d_out]
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(linear.weight)
            self.linear.append(linear)
    
    def forward(self,features):
        x = features
        for linear in self.linear:
            x = linear(x)
        return x
