# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Encode(nn.Module):
    def __init__(self):
        super(Encode,self).__init__()
        self.lstm=nn.LSTM(10,3)
    def forward(self,x):
        return self.lstm(x,None)

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.embed=nn.Embedding(2,5)
    def forward(self,x):
        return self.embed(x)

class attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(attention,self).__init__()
        self.embed_dim=embed_dim
        self.num_head=num_heads
        self.head_dim=int(embed_dim/num_heads)
        self.all_head_dim=embed_dim
        self.qkv=nn.Linear(embed_dim,self.all_head_dim*3)
        self.scale=self.head_dim**-0.5
        self.softmax=nn.Softmax(-1)
        self.proj=nn.Linear(self.all_head_dim,self.embed_dim)
    def tran(self,x):
        #[B,N,embed]
        new_shape=list(x.shape[:-1])+[self.num_head,self.head_dim]
        print("1",new_shape)
        # [B,N,num_head,head_dim]
        x=x.reshape(new_shape)
        x=x.permute([0,2,1,3])
        x=x.contiguous()
        # [B,num_head,N,head_dim]
        print("2", x.shape)
        return x
    def forward(self,x):
        B,N,_=x.shape
        qkv=self.qkv(x).chunk(3,-1)
        print("a",qkv[0].shape)
        q,k,v=map(self.tran,qkv)
        k=k.permute([0,1,3,2])
        #[B, num_head, N, head_dim]
        #[B, num_head, head_dim,N]
        print("3",q.shape,'4',k.shape)
        attn=torch.matmul(q,k)
        print("att",attn.shape)

        attn=self.scale*attn
        attn=self.softmax(attn)
        #[B, num_head, N,N]
        out=torch.matmul(attn,v)
        #[B, num_head, N,N]*[# [B,num_head,N,head_dim]]
        out=out.permute([0,2,1,3])
        out=out.contiguous()
        print('5',out.shape)
        out=out.reshape([B,N,-1])
        out=self.proj(out)
        return out

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    AT=attention(96,4)

    x=torch.rand(6,10,96)
    aa=AT(x)

    print(aa.shape)



