from srcs.model.bd_modules import Conv, ResBlock, MLP, CUnet,DepthwiseSeparableConv
import torch.nn as nn
import torch
from srcs.model.bd_utils import PositionalEncoding

class BDNeRV_RC(nn.Module):
    # recursive frame reconstruction
    def __init__(self):
        super(BDNeRV_RC, self).__init__()
        # params
        n_colors = 1
        n_resblock = 12
        n_feats = 32
        kernel_size = 3
        padding = 1


        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [14*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        # mlp_dim_list = [2*pos_l, 512, n_feats] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)
        
        # 新增的最终输出处理层 
        # self.final_conv = nn.Sequential(
        #     Conv_D(1, 32, kernel_size=3, dilation=255, padding=255),  
        #     Conv_D(32, 64, kernel_size=3, dilation=255, padding=255),   
        #     Conv_D(64, 128, kernel_size=3, dilation=255, padding=255),   
        #     Conv_D(128, 1, kernel_size=1)
        #     )
        self.final_conv = nn.Sequential(
            DepthwiseSeparableConv(4096,4096,kernel_size=1, padding=0),  # 显式禁止padding
            DepthwiseSeparableConv(4096,4096,kernel_size=1, padding=0)
        )
        

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)
        # 新增积分时间编码参数
        self.pe_int_time = PositionalEncoding(pe_embed_b=1.0, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, time_idx,  ce_code, temp_huanjing,components):
            # 设备同步
            ratio = 0
            device = ce_blur.device
            time_idx = time_idx.float().to(device)
            temp_huanjing = temp_huanjing.float().to(device)
            components = components.float().to(device)
            # 联合嵌入温度和时间信息
            t_pe_ = []
            # print(temp_huanjing.shape) #[4,1]
            temp_huanjing = torch.cat([temp_huanjing, components], dim=1)
            # print(temp_huanjing.shape) #[4,6]
            #     # 将 int_time 转换为张量
            # int_time = torch.tensor(int_time, device=ce_blur.device)

            # 扩展 int_time 的维度以匹配 batch size
            batch_size = ce_blur.size(0)
            # print(temp_huanjing.shape)
            # temp_huanjing = temp_huanjing.unsqueeze(0).repeat(batch_size,1)  # [B,1]
            # time_idx = time_idx.unsqueeze(0).repeat(batch_size,1) # [B,1]
            # print(ce_blur.shape,time_idx,temp_huanjing)
            # print(f"temp_idx shape: {time_idx.shape}")   # 应为  [B,1]
            # print(f"int_time shape: {temp_huanjing.shape}")   # 应为  [B,1]
            # print(f"ce_code shape: {ce_code.shape}")     # 应与上述一致
            # print(ee)
            for t_idx, temp_vector, code in zip(time_idx, temp_huanjing, ce_code):
                # 温度嵌入
                time_pe = self.pe_t(t_idx).squeeze(-1)  
                # # 积分时间嵌入
                # int_pe = self.pe_int_time(int_t).squeeze(-1) 
                # 温度参数逐个编码
                temp_pes = []
                for temp_val in temp_vector:  # 遍历每个温度值（共6个）
                    pe = self.pe_int_time(temp_val)  # [1, pos_l, 1]
                    temp_pes.append(pe)
                # 合并温度编码
                combined_temp_pe = torch.cat(temp_pes, dim=1)  # [1, 6*pos_l, 1]
                # 联合编码
                joint_pe = torch.cat([time_pe, combined_temp_pe], dim=1)  # [1,7*pos_l,1]
                joint_pe = joint_pe.squeeze(-1)  # [1,7*pos_l]
                # 联合编码
                # print(temp_idx,int_time)
                # print(time_pe.shape, combined_temp_pe.shape) #[1,160],[1,960]
                # joint_pe = torch.cat([time_pe, int_pe], dim=0)  
                t_pe_.append(joint_pe * (2 * code - 1))

            t_pe = torch.cat(t_pe_, dim=1)  # [frame_num, pos_l*2]
            # 在 bd_model.py 的 forward() 中添加调试代码
            # print(f"t_pe shape: {t_pe.shape}")        # 输入维度
            # print(f"embed_mlp weight shape: {self.embed_mlp[0].weight.shape}")
            t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4 * 2]
            
            # ce_blur feature
            ce_feature = self.feature(ce_blur)  # [b, c, h, w]

            # main body
            output_list = []
            # print(len(time_idx))
            # print(ee)
            for k in range(1):
                if k==0 :
                    main_feature = ce_feature
                else:
                    # since k=2, cat pre-feature with ce_feature as input feature
                    cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                    main_feature = self.catfusion(cat_feature)
                feat_out_k = self.mainbody(main_feature, t_embed[k])
                output_k1 = self.out(feat_out_k)
                B, C, H, W = output_k1.shape  # 获取原始形状
                start_h = (H - 64) // 2      # 计算高度起始索引
                start_w = (W - 64) // 2      # 计算宽度起始索引
                output_k_crop = output_k1[:, :, start_h:start_h+64, start_w:start_w+64]
                B1,C1,H1,W1=output_k_crop.shape


                output_k_p = output_k_crop.reshape(B1,H1*W1,1,1)
                output_k_p = self.final_conv(output_k_p)

                output_k2 = output_k_p.reshape(B1,C1,H1,W1)

                output_k =  ( 1 - ratio) *output_k2
                # output_k = self.out(main_feature)
                output_list.append(output_k)

            output = torch.stack(output_list, dim=1)
            # output = torch.clamp(output, 0, 1)

            return output
   