import torch
import torch.nn as nn
import numpy as np
def agg_tensor(tensor, neigh_w, mlp):
    forward_roll=torch.roll(tensor, shifts=1, dims=0)
    backward_roll=torch.roll(tensor, shifts=-1, dims=0)
    neigh_tensor=(mlp(forward_roll)+mlp(backward_roll))
    return neigh_w*neigh_tensor+(1-2*neigh_w)*tensor#fixed bug, 20220926

class bignet2(nn.Module):
    def __init__(self, device='cpu', brands=6, cor_init_rand=0):
        super(bignet2, self).__init__()
        self.device=device
        self.brands=brands
        self.neigh_w=0.25
        self.crv_neigh_dist=2

        self.crv_len=8
        self.crv_embed_len=self.crv_len*(self.crv_neigh_dist+1)

        self.act=nn.LeakyReLU()

        #now blank, will fill with agg tensor later
        # curve_tensor_agg=torch.zeros((crv_tot, crv_len*(neigh_dist+1)))

        #can add linear layer, maybe
        self.crv_mlp=nn.Linear(self.crv_len,self.crv_len) #for ctrl pt to ctrl pt

        self.crv_w=32
        self.crv_features_final=24

        self.crv_lv1=nn.Linear(self.crv_embed_len,self.crv_w)#24x24
        self.crv_lv2=nn.Linear(self.crv_w,self.crv_features_final)
        
        self.group_len=24

        #make group tensor go thru some mlp
        self.inter_mlp1=nn.Linear(self.crv_features_final, self.crv_features_final)#24x24
        self.inter_mlp2=nn.Linear(self.crv_features_final, self.group_len)#24x24

        #now: aggragate from group to image
        self.group_neigh_dist=2

        # group_tot, group_len=group_tensor.shape
        self.group_embed_len=self.group_len*(self.group_neigh_dist+1)

        self.group_w1=24
        self.group_w2=24
        self.group_features_final=24

        self.group_lv1=nn.Linear(self.group_embed_len,self.group_w1)#36x24
        self.group_lv2=nn.Linear(self.group_w1,self.group_w2)
        self.group_lv3=nn.Linear(self.group_w2,self.group_features_final)

        #first:map cor to the right size
        self.cor_par=5
        self.cor_mlp=nn.Linear(self.cor_par,self.group_len)#5x12
        '''
        initialize!!!
        cor_mlp.weight.data.fill_(1)
        cor_mlp.bias.data.fill_(1)
        '''
        if cor_init_rand==0:
            self.cor_mlp.weight.data.fill_(1)
            self.cor_mlp.bias.data.fill_(1)

        self.img_w1=18
        self.img_w2=12
        # self.brands
        # self.brands=6

        self.img_lv1=nn.Linear(self.group_features_final,self.img_w1)
        self.img_lv2=nn.Linear(self.img_w1,self.img_w2)
        self.img_lv3=nn.Linear(self.img_w2,self.brands)

    def forward(self, curve_tensor, curve_label, cors_batch):
        
        curve_tensor=curve_tensor.to(self.device)
        crv_starts=np.where(curve_label[:,1]==1)[0]
        crv_ends=np.roll(crv_starts, shift=-1)
        group_tot=crv_starts.shape[0]

        group_tensor=torch.zeros((group_tot, self.crv_features_final)).to(self.device)
        
        group_starts=np.where(curve_label[:,0]==1)[0]
        group_ends=np.roll(group_starts, shift=-1)
        img_tot=group_starts.shape[0]

        img_tensor=torch.zeros((img_tot, self.group_features_final)).to(self.device)

        for idx, (start, end) in enumerate(zip(crv_starts,crv_ends)):#for every curve in a group
            if end==0:#the last curve
                end=None
            group_agg_start=curve_tensor[start:end]
            group_agg_before=curve_tensor[start:end]
            for i in range(self.crv_neigh_dist):
                group_agg_after=agg_tensor(group_agg_before, self.neigh_w, self.crv_mlp)#using mlp, can be unstable
                group_agg_start=torch.hstack((group_agg_start,group_agg_after))
                group_agg_before=group_agg_after

            group_agg_final=self.act(self.crv_lv2(self.act(self.crv_lv1(group_agg_start))))
            group_tensor[idx]=torch.mean(group_agg_final, axis=0)#nx24

        group_tensor_final=self.act(self.inter_mlp2(self.act(self.inter_mlp1(group_tensor))))

        for idx, (start, end) in enumerate(zip(group_starts,group_ends)):#for every group in a pic
#             print(start,end)#還要找到start, end是第幾個為1的curve
            group_start=int(np.where(np.where(curve_label[:,1]==1)[0]==start)[0])#the "group_start"th first curve
            if end==0:#the last group
                group_end=None
            else:
                group_end=int(np.where(np.where(curve_label[:,1]==1)[0]==end)[0])
            #new: product then sum
            cor_mat=torch.FloatTensor(cors_batch[idx]).to(self.device)
            cor_mat_adapted=self.act(self.cor_mlp(cor_mat))
            #old: sum then product, lose 1 degree of freedom
            # cor_mat1d=torch.sum(torch.FloatTensor(cors_batch[idx]).to(self.device), axis=1)#把同分母不同分子加起來
            # cor_mat_adapted=self.act(self.cor_mlp(cor_mat1d))#groupx12

            img_agg_start=group_tensor_final[group_start:group_end]
            img_agg_before=group_tensor_final[group_start:group_end]

            for i in range(self.group_neigh_dist):
                #new: product then sum
                img_agg_after=torch.sum(img_agg_before*cor_mat_adapted, axis=1)
                #old: sum then product, lose 1 degree of freedom
                # img_agg_after=img_agg_before*cor_mat_adapted
                img_agg_start=torch.hstack((img_agg_start,img_agg_after))
                img_agg_before=img_agg_after

            img_agg_final=self.act(self.group_lv3(self.act(self.group_lv2(self.act(self.group_lv1(img_agg_start))))))#groupx24

            img_tensor[idx]=torch.mean(img_agg_final, axis=0)#1x24->batchx24

        '''no softmax!!!'''            
        label_final= self.img_lv3(self.act(self.img_lv2(self.act(self.img_lv1(img_tensor)))))
        return nn.Sigmoid()(label_final)
    
class bignet2_latent1(nn.Module):
    def __init__(self, device='cpu', brands=6, cor_init_rand=0):
        super(bignet2_latent1, self).__init__()
        self.device=device
        self.brands=brands
        self.neigh_w=0.25
        self.crv_neigh_dist=2

        self.crv_len=8
        self.crv_embed_len=self.crv_len*(self.crv_neigh_dist+1)

        self.act=nn.LeakyReLU()

        #now blank, will fill with agg tensor later
        # curve_tensor_agg=torch.zeros((crv_tot, crv_len*(neigh_dist+1)))

        #can add linear layer, maybe
        self.crv_mlp=nn.Linear(self.crv_len,self.crv_len) #for ctrl pt to ctrl pt

        self.crv_w=32
        self.crv_features_final=24

        self.crv_lv1=nn.Linear(self.crv_embed_len,self.crv_w)#24x24
        self.crv_lv2=nn.Linear(self.crv_w,self.crv_features_final)
        
        self.group_len=24

        #make group tensor go thru some mlp
        self.inter_mlp1=nn.Linear(self.crv_features_final, self.crv_features_final)#24x24
        self.inter_mlp2=nn.Linear(self.crv_features_final, self.group_len)#24x24

        #now: aggragate from group to image
        self.group_neigh_dist=2

        # group_tot, group_len=group_tensor.shape
        self.group_embed_len=self.group_len*(self.group_neigh_dist+1)

        self.group_w1=24
        self.group_w2=24
        self.group_features_final=24

        self.group_lv1=nn.Linear(self.group_embed_len,self.group_w1)#36x24
        self.group_lv2=nn.Linear(self.group_w1,self.group_w2)
        self.group_lv3=nn.Linear(self.group_w2,self.group_features_final)

        #first:map cor to the right size
        self.cor_par=5
        self.cor_mlp=nn.Linear(self.cor_par,self.group_len)#5x12
        '''
        initialize!!!
        cor_mlp.weight.data.fill_(1)
        cor_mlp.bias.data.fill_(1)
        '''
        if cor_init_rand==0:
            self.cor_mlp.weight.data.fill_(1)
            self.cor_mlp.bias.data.fill_(1)

        self.img_w1=18
        self.img_w2=12

        self.img_lv1=nn.Linear(self.group_features_final,self.img_w1)
        self.img_lv2=nn.Linear(self.img_w1,self.img_w2)
        self.img_lv3=nn.Linear(self.img_w2,self.brands)

    def forward(self, curve_tensor, curve_label, cors_batch):
        
        curve_tensor=curve_tensor.to(self.device)
        crv_starts=np.where(curve_label[:,1]==1)[0]
        crv_ends=np.roll(crv_starts, shift=-1)
        group_tot=crv_starts.shape[0]

        group_tensor=torch.zeros((group_tot, self.crv_features_final)).to(self.device)
        
        group_starts=np.where(curve_label[:,0]==1)[0]
        group_ends=np.roll(group_starts, shift=-1)
        img_tot=group_starts.shape[0]

        img_tensor=torch.zeros((img_tot, self.group_features_final)).to(self.device)

        for idx, (start, end) in enumerate(zip(crv_starts,crv_ends)):#for every curve in a group
            if end==0:
                end=None
            group_agg_start=curve_tensor[start:end]
            group_agg_before=curve_tensor[start:end]
            for i in range(self.crv_neigh_dist):
                group_agg_after=agg_tensor(group_agg_before, self.neigh_w, self.crv_mlp)#using mlp, can be unstable
                group_agg_start=torch.hstack((group_agg_start,group_agg_after))
                group_agg_before=group_agg_after

            group_agg_final=self.act(self.crv_lv2(self.act(self.crv_lv1(group_agg_start))))
            group_tensor[idx]=torch.mean(group_agg_final, axis=0)#nx24

        group_tensor_final=self.act(self.inter_mlp2(self.act(self.inter_mlp1(group_tensor))))

        for idx, (start, end) in enumerate(zip(group_starts,group_ends)):#for every group in a pic
#             print(start,end)#還要找到start, end是第幾個為1的curve
            group_start=int(np.where(np.where(curve_label[:,1]==1)[0]==start)[0])#the "group_start"th first curve
            if end==0:#the last group
                group_end=None
            else:
                group_end=int(np.where(np.where(curve_label[:,1]==1)[0]==end)[0])
            #new: product then sum
            cor_mat=torch.FloatTensor(cors_batch[idx]).to(self.device)
            cor_mat_adapted=self.act(self.cor_mlp(cor_mat))
            #old: sum then product, lose 1 degree of freedom
            # cor_mat1d=torch.sum(torch.FloatTensor(cors_batch[idx]).to(self.device), axis=1)#把同分母不同分子加起來
            # cor_mat_adapted=self.act(self.cor_mlp(cor_mat1d))#groupx12

            img_agg_start=group_tensor_final[group_start:group_end]
            img_agg_before=group_tensor_final[group_start:group_end]

            for i in range(self.group_neigh_dist):
                #new: product then sum
                img_agg_after=torch.sum(img_agg_before*cor_mat_adapted, axis=1)
                #old: sum then product, lose 1 degree of freedom
                # img_agg_after=img_agg_before*cor_mat_adapted
                img_agg_start=torch.hstack((img_agg_start,img_agg_after))
                img_agg_before=img_agg_after

            img_agg_final=self.act(self.group_lv3(self.act(self.group_lv2(self.act(self.group_lv1(img_agg_start))))))#groupx24

            img_tensor[idx]=torch.mean(img_agg_final, axis=0)#1x24->batchx24

        '''no softmax!!!'''            
        label_final= self.img_lv3(self.act(self.img_lv2(self.act(self.img_lv1(img_tensor)))))
        return self.img_lv2(self.act(self.img_lv1(img_tensor))), nn.Sigmoid()(label_final)
    
    
class bignet2_latent2(nn.Module):
    def __init__(self, device='cpu', brands=6, cor_init_rand=0):
        super(bignet2_latent2, self).__init__()
        self.device=device
        self.brands=brands
        self.neigh_w=0.25
        self.crv_neigh_dist=2

        self.crv_len=8
        self.crv_embed_len=self.crv_len*(self.crv_neigh_dist+1)

        self.act=nn.LeakyReLU()

        #now blank, will fill with agg tensor later
        # curve_tensor_agg=torch.zeros((crv_tot, crv_len*(neigh_dist+1)))

        #can add linear layer, maybe
        self.crv_mlp=nn.Linear(self.crv_len,self.crv_len) #for ctrl pt to ctrl pt

        self.crv_w=32
        self.crv_features_final=24

        self.crv_lv1=nn.Linear(self.crv_embed_len,self.crv_w)#24x24
        self.crv_lv2=nn.Linear(self.crv_w,self.crv_features_final)
        
        self.group_len=24

        #make group tensor go thru some mlp
        self.inter_mlp1=nn.Linear(self.crv_features_final, self.crv_features_final)#24x24
        self.inter_mlp2=nn.Linear(self.crv_features_final, self.group_len)#24x24

        #now: aggragate from group to image
        self.group_neigh_dist=2

        # group_tot, group_len=group_tensor.shape
        self.group_embed_len=self.group_len*(self.group_neigh_dist+1)

        self.group_w1=24
        self.group_w2=24
        self.group_features_final=24

        self.group_lv1=nn.Linear(self.group_embed_len,self.group_w1)#36x24
        self.group_lv2=nn.Linear(self.group_w1,self.group_w2)
        self.group_lv3=nn.Linear(self.group_w2,self.group_features_final)

        #first:map cor to the right size
        self.cor_par=5
        self.cor_mlp=nn.Linear(self.cor_par,self.group_len)#5x12
        '''
        initialize!!!
        cor_mlp.weight.data.fill_(1)
        cor_mlp.bias.data.fill_(1)
        '''
        if cor_init_rand==0:
            self.cor_mlp.weight.data.fill_(1)
            self.cor_mlp.bias.data.fill_(1)

        self.img_w1=18
        self.img_w2=12

        self.img_lv1=nn.Linear(self.group_features_final,self.img_w1)
        self.img_lv2=nn.Linear(self.img_w1,self.img_w2)
        self.img_lv3=nn.Linear(self.img_w2,self.brands)

    def forward(self, curve_tensor, curve_label, cors_batch):
        
        curve_tensor=curve_tensor.to(self.device)
        crv_starts=np.where(curve_label[:,1]==1)[0]
        crv_ends=np.roll(crv_starts, shift=-1)
        group_tot=crv_starts.shape[0]

        group_tensor=torch.zeros((group_tot, self.crv_features_final)).to(self.device)
        
        group_starts=np.where(curve_label[:,0]==1)[0]
        group_ends=np.roll(group_starts, shift=-1)
        img_tot=group_starts.shape[0]

        img_tensor=torch.zeros((img_tot, self.group_features_final)).to(self.device)

        for idx, (start, end) in enumerate(zip(crv_starts,crv_ends)):#for every curve in a group
            if end==0:
                end=None
            group_agg_start=curve_tensor[start:end]
            group_agg_before=curve_tensor[start:end]
            for i in range(self.crv_neigh_dist):
                group_agg_after=agg_tensor(group_agg_before, self.neigh_w, self.crv_mlp)#using mlp, can be unstable
                group_agg_start=torch.hstack((group_agg_start,group_agg_after))
                group_agg_before=group_agg_after

            group_agg_final=self.act(self.crv_lv2(self.act(self.crv_lv1(group_agg_start))))
            group_tensor[idx]=torch.mean(group_agg_final, axis=0)#nx24

        group_tensor_final=self.act(self.inter_mlp2(self.act(self.inter_mlp1(group_tensor))))

        for idx, (start, end) in enumerate(zip(group_starts,group_ends)):#for every group in a pic
#             print(start,end)#還要找到start, end是第幾個為1的curve
            group_start=int(np.where(np.where(curve_label[:,1]==1)[0]==start)[0])#the "group_start"th first curve
            if end==0:#the last group
                group_end=None
            else:
                group_end=int(np.where(np.where(curve_label[:,1]==1)[0]==end)[0])
            #new: product then sum
            cor_mat=torch.FloatTensor(cors_batch[idx]).to(self.device)
            cor_mat_adapted=self.act(self.cor_mlp(cor_mat))
            #old: sum then product, lose 1 degree of freedom
            # cor_mat1d=torch.sum(torch.FloatTensor(cors_batch[idx]).to(self.device), axis=1)#把同分母不同分子加起來
            # cor_mat_adapted=self.act(self.cor_mlp(cor_mat1d))#groupx12

            img_agg_start=group_tensor_final[group_start:group_end]
            img_agg_before=group_tensor_final[group_start:group_end]

            for i in range(self.group_neigh_dist):
                #new: product then sum
                img_agg_after=torch.sum(img_agg_before*cor_mat_adapted, axis=1)
                #old: sum then product, lose 1 degree of freedom
                # img_agg_after=img_agg_before*cor_mat_adapted
                img_agg_start=torch.hstack((img_agg_start,img_agg_after))
                img_agg_before=img_agg_after

            img_agg_final=self.act(self.group_lv3(self.act(self.group_lv2(self.act(self.group_lv1(img_agg_start))))))#groupx24

            img_tensor[idx]=torch.mean(img_agg_final, axis=0)#1x24->batchx24
        
        groups_final = self.img_lv3(self.act(self.img_lv2(self.act(self.img_lv1(img_agg_final)))))
        return nn.Sigmoid()(groups_final)
        '''no softmax!!!'''            
        # label_final= self.img_lv3(self.act(self.img_lv2(self.act(self.img_lv1(img_tensor)))))
        # return img_tensor, nn.Sigmoid()(label_final)