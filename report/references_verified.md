# Verified references

Checked on 2026-10-13 against Crossref (api.crossref.org) and arXiv. IEEE Xplore blocks scripted access,
so DOIs were confirmed through Crossref. "Checked in text" means the PDF was read for the stated detail.

## Anchor papers

| Key | Reference | DOI / link | Notes |
|---|---|---|---|
| STAGNet | V. Vipulananthan, K. Mohottala, K. Chinthana, N. Paramulla, et al., "STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation," *IEEE Access*, vol. 13, pp. 213769–213779, 2025. arXiv:2508.15216 | 10.1109/ACCESS.2025.3645127 | Checked in text: standard cross-entropy loss (Eq. 7); mTTA = mean TTA over thresholds α; CCD excluded for video quality bias; future work: segmentation for tighter vehicle delineation, depth / camera intrinsics for inter-vehicle distance. Code: github.com/Vipooshan1998/STAGNet |
| DoTA | Y. Yao, X. Wang, M. Xu, Z. Pu, Y. Wang, E. Atkins, D. J. Crandall, "DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos," *IEEE TPAMI*, vol. 45, no. 1, pp. 444–459, 2023. arXiv:2004.03044 (titled "When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos") | 10.1109/TPAMI.2022.3150763 | TPAMI title differs from the arXiv title |
| DAD | F.-H. Chan, Y.-T. Chen, Y. Xiang, M. Sun, "Anticipating Accidents in Dashcam Videos," ACCV 2016, LNCS vol. 10114, pp. 136–153, Springer, 2017 | 10.1007/978-3-319-54190-7_9 | Project page: 620 positive / 1130 negative clips, 100 frames (5 s), accident in the last 10 frames of positives, 1284 train / 466 test |
| CCD (UString) | W. Bao, Q. Yu, Y. Kong, "Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning," ACM MM 2020, pp. 2682–2690 | 10.1145/3394171.3413827 | CCD repo: 1500 crash + 3000 normal (BDD100K) videos, 50 frames at 10 fps |
| Graph(Graph) | N. Thakur, P. Gouripeddi, B. Li, "Graph(Graph): A Nested Graph-Based Framework for Early Accident Anticipation," WACV 2024, pp. 7518–7526 | 10.1109/WACV57701.2024.00736 | Code and splits: github.com/thakurnupur/Graph-Graph |
| Survey | J. Fang, J. Qiao, J. Xue, Z. Li, "Vision-Based Traffic Accident Detection and Anticipation: A Survey," *IEEE TCSVT*, vol. 34, no. 4, pp. 1983–1999, 2024 | 10.1109/TCSVT.2023.3307655 | |

## Additional post-2022 IEEE papers (confirmed via Crossref)

| Reference | DOI |
|---|---|
| M. M. Karim, Y. Li, R. Qin, Z. Yin, "A Dynamic Spatial-Temporal Attention Network for Early Anticipation of Traffic Accidents" (DSTA), *IEEE T-ITS*, vol. 23, no. 7, pp. 9590–9600, 2022 | 10.1109/TITS.2022.3155613 |
| H. Li, J. Fang, J. Xue, et al., "Cognitive Traffic Accident Anticipation," *IEEE ITS Magazine*, vol. 16, no. 5, pp. 17–32, 2024 | 10.1109/MITS.2024.3378460 |
| Patera, Chen, Fang, et al., "Spatio-Temporal Adaptation With Dilated Neighbourhood Attention For Accident Anticipation," IEEE ICIP 2024, pp. 2452–2458 | 10.1109/ICIP51287.2024.10647316 |
| Fang, Li, Zheng, Yu, et al., "EQ-TAA: Equivariant Traffic Accident Anticipation via Diffusion-Based Accident Video Synthesis," *IEEE TMM*, vol. 27, pp. 8628–8643, 2025 | 10.1109/TMM.2025.3607808 |

Author lists above are truncated to what Crossref returned; full author lists will be taken from the DOI record when the report is written.

## Still to verify before use
- The exact form of the exponential anticipation loss (Chan et al. 2016; as used in UString / DSTA code).
- xLSTM (only if the stretch ablation is attempted).
