import torch
import numpy as np 

def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "min": torch.min(v, 0)[0],
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "10%": v.kthvalue(int(len(v) * 0.10), dim=0)[0],
            "15%": v.kthvalue(int(len(v) * 0.15), dim=0)[0],
            "20%": v.kthvalue(int(len(v) * 0.20), dim=0)[0],
            "25%": v.kthvalue(int(len(v) * 0.25), dim=0)[0],
            "30%": v.kthvalue(int(len(v) * 0.30), dim=0)[0],
            "35%": v.kthvalue(int(len(v) * 0.35), dim=0)[0],
            "40%": v.kthvalue(int(len(v) * 0.40), dim=0)[0],
            "45%": v.kthvalue(int(len(v) * 0.45), dim=0)[0],
            "50%": v.kthvalue(int(len(v) * 0.50), dim=0)[0],
            "55%": v.kthvalue(int(len(v) * 0.55), dim=0)[0],
            "60%": v.kthvalue(int(len(v) * 0.60), dim=0)[0],
            "65%": v.kthvalue(int(len(v) * 0.65), dim=0)[0],
            "70%": v.kthvalue(int(len(v) * 0.70), dim=0)[0],
            "75%": v.kthvalue(int(len(v) * 0.75), dim=0)[0],
            "80%": v.kthvalue(int(len(v) * 0.80), dim=0)[0],
            "85%": v.kthvalue(int(len(v) * 0.85), dim=0)[0],
            "90%": v.kthvalue(int(len(v) * 0.90), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            "max": torch.max(v, 0)[0],
        }
    return site_stats

def check_Z(mean, std):
    site_stats = {
        "mean": mean,
        "std": std,
        "min": mean-2.58*std,
        "2.5%":mean-1.96*std,
        "5%": mean-1.64*std,
        "10%": mean-1.29*std,
        "15%": mean-1.04*std,
        "20%": mean-0.85*std,
        "25%": mean-0.68*std,
        "30%":mean-0.53*std,
        "35%": mean-0.39*std,
        "40%": mean-0.26*std,
        "45%": mean-0.13*std,
        "50%": mean-0*std,
        "55%": mean+0.13*std,
        "60%": mean+0.26*std,
        "65%": mean+0.39*std,
        "70%": mean+0.53*std,
        "75%": mean+0.68*std,
        "80%": mean+0.85*std,
        "85%": mean+1.04*std,
        "90%": mean+1.29*std,
        "95%": mean+1.64*std,
        "97.5%": mean+1.96*std,
        "max": mean+2.58*std,
    }
    return site_stats

def eval_uncertainty(labels, preds, quantiles={"5%":0.05, "25%": 0.25, "50%":0.5, "75%":0.75, "95%":0.95}):
    N = len(labels)
    cal, sharp = 0.0, 0.0
    for k, pj in quantiles.items():
        y_pj= preds[k]
        count = sum([1 for i in range(N) if labels[i] <= y_pj[i]])
        pj_hat = count/N
        cal += pow((pj-pj_hat), 2)
    
    mean, std = preds['mean'], preds['std']
    labels = torch.tensor(labels, dtype=torch.float)
    nldp = torch.mean((pow((mean-labels), 2) / pow(std, 2) + torch.log(pow(std, 2)))/2, 0)
    sharp = torch.mean(preds['std'], 0)
    dis = torch.std(preds['std'], 0)
    uncertainty_metrics = {"cal": round(cal, 3), "sha": round(sharp.item(), 3), 
                           "dis": round(dis.item(), 3), "nldp": round(nldp.item(), 3)}
    for k,v in uncertainty_metrics.items():
        print(k, v)
    return uncertainty_metrics