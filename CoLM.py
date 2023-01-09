import torch
import torch.nn.functional as F


def CC_loss(proj, label, prob, temp, hard_sample, negatives_nums):
    classes_nums = label.shape[1]
    proj = proj.permute(0, 2, 3, 1)
    pix_feats_all = []
    pix_hard_all = []
    pix_num_all = []
    class_mean_all = []
    
    for i in range(classes_nums):
        valid_pixel = label[:, i] 
        if valid_pixel.sum() == 0:  
            continue

        prob_pix = prob[:, i, :, :] 
        prob_pix_val = prob_pix * valid_pixel.bool()
        prob_pix_val =  torch.where(prob_pix_val != 0, prob_pix_val, torch.ones_like(prob_pix_val))
        prob_pix_mask = hard_sampler(prob_pix_val,hard_sample)
        class_mean_all.append(torch.mean(proj[valid_pixel.bool()], dim=0, keepdim=True)) 
        pix_feats_all.append(proj[valid_pixel.bool()]) 
        pix_hard_all.append(proj[prob_pix_mask])
        pix_num_all.append(int(valid_pixel.sum().item())) 

    # Compute pixel contrastive loss
    if len(pix_num_all) <= 1:  
        return torch.tensor(0.0)
    else:
        loss_CC = torch.tensor(0.0)
        class_mean = torch.cat(class_mean_all) 
        valid_class = len(pix_num_all) 
        batch_class = torch.arange(valid_class)

        for i in range(valid_class):
            if len(pix_hard_all[i]) > 0: 
                pix_hard_all[i] = pix_hard_all[i]
            else:  
                continue
            with torch.no_grad():    
        
                class_mask = torch.cat(([batch_class[i:], batch_class[:i]])) 
                pix_feats_other_class = [pix_feats_all[i] for i in class_mask[1:]] 
                pix_feats_other_class_mean = [class_mean_all[i] for i in class_mask[1:]] 
                idx_negatives_set = []
                for j in range(len(pix_feats_other_class_mean)):
                    distance_pix = torch.cosine_similarity(pix_feats_other_class[j], pix_feats_other_class_mean[j], dim=1)
                    nums = min(negatives_nums, pix_feats_other_class[j].shape[0])
                    _, position_pix = torch.topk(distance_pix, nums, dim=0, sorted=False, out=None)
                    idx_negatives_set.append(position_pix)     

                # Negative samples features
                negative_feat_all = torch.cat(pix_feats_all[i+1:] + pix_feats_all[:i]) 
                idx_negatives_set_idx = []
                for j in range(len(idx_negatives_set)):
                    idx_negatives_set_idx += idx_negatives_set[j].tolist()
                
                negative_feat = negative_feat_all[idx_negatives_set_idx] # 

            matrix_posi = torch.cosine_similarity(pix_hard_all[i], class_mean[i].unsqueeze(0)) 
            similarity_posi = torch.exp(torch.mean(matrix_posi) / temp)

            matrix_neg = torch.cosine_similarity(pix_hard_all[i].unsqueeze(1), negative_feat.unsqueeze(0), dim=2)
            similarity_neg = torch.exp((torch.mean(matrix_neg)) / temp) 
            loss_CC = loss_CC + (-torch.log(similarity_posi / (similarity_neg + similarity_posi))) 
        return loss_CC / valid_class
            
def hard_sampler(pix_set, hard_sample):
    smaller, _= pix_set.topk(k=hard_sample,dim=2, largest=False)
    pix_s =torch.max(smaller,dim=-1).values
    pix_s = pix_s.unsqueeze(-1).repeat(1,1,pix_set.shape[-1])
    ge=torch.ge(pix_s,pix_set) 
    zero = torch.zeros_like(pix_set)
    hard_sample = torch.where(ge,pix_set,zero)
    hard_mask_1 = hard_sample > 0 
    hard_mask_2 = hard_sample < 1
    hard_mask = hard_mask_1 * hard_mask_2 
    return hard_mask

def SCC(features, features_aug):

    valid_n = features.shape[0]*features.shape[2]*features.shape[3]
    c_sim = torch.cosine_similarity(features, features_aug, dim=1)
    loss_SCC = F.cross_entropy(c_sim, torch.ones((features.shape[0],features.shape[3])).long().cuda()) / valid_n
    return loss_SCC

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    inputs = torch.tensor(inputs,dtype = torch.int64)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
