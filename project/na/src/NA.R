# install.packages("psych")
# install.packages("bootnet")
# install.packages("qgraph")
# install.packages("networktools")
# install.packages("mgm")
library(psych)
library(bootnet)
library(qgraph)
library(ggplot2)
library(networktools)
library(mgm)
bfi_clean <- na.omit(bfi[, 1:25]) # 删除带有 NA 值的数据行

############################# 1. 分组 #############################
comm_labels <- c(
  rep("Agreeableness", 5),    # A1-A5
  rep("Conscientiousness", 5),# C1-C5
  rep("Extraversion", 5),     # E1-E5
  rep("Neuroticism", 5),      # N1-N5
  rep("Openness", 5)          # O1-O5
)

names(comm_labels) <- colnames(bfi_clean) # 指定


############################# 2. 计算可预测性 #############################
network <- as.matrix(bfi_clean) # mgm 函数需要使用矩阵数据
p <- ncol(network) # 用于指定数据框的列数
# mgm 在拟合模型后，会自动计算每个节点的 out-of-sample 预测准确性（通过留一法或交叉验证）。
fit_obj <- mgm(data = network, 
               type = rep("g", p),  # "g"表示高斯（连续）变量
               level = rep(1, p),   # 对于连续变量设为1
               lambdaSel = "CV",    # 使用交叉验证选择lambda
               ruleReg = "OR",      # 正则化规则
               pbar = TRUE)         # 打开进度条
# 利用已拟合的 mgm 模型，计算每个节点被其余所有节点预测时的拟合优度(虽然mgm算了但不会返回该值)。
pred_obj <- predict(object = fit_obj,
                    data = network,
                    errorCon = 'R2')
pred_obj$error # 可预测性计算结果


############################# 3. 绘制带有可预测性的网络图 #############################
plot_network <- plot(net_bfi,
                     layout = "spring",
                     groups = comm_labels,
                     label.cex = 1, # 调整节点中标签的字体大小
                     label.color = 'black', # 调整节点中标签的字体颜色
                     negDashed = T, # 负相关的边为虚线
                     legend=F,
                     # nodeNames = items,
                     legend.cex = 0.32, # 图例的字体大小
                     legend.mode = 'style1', # 图例的风格，默认是'style1'
                     pie = pred_obj$error[,2]) # 显示可预测性的环


############################# 4. 中心性指标  #############################
centralityPlot(net_bfi, include = "all",  # 绘图中心性指标的图
               orderBy = "ExpectedInfluence", scale = c("z-scores"))
                # 按照 ExpectedInfluence 排序，输出z分数
centrality_auto(net_bfi)


############################# 5. 中心性指标的稳定性检验  #############################
boot_net <- bootnet(net_bfi,
                    nBoots = 100, # 重抽样次数
                    nCores = 8, # 参与运算的CPU核心数量
                    statistics = c('strength','expectedInfluence',
                                   'betweenness','closeness',"edge"))
# 绘制边权重的 95% 置信区间
plot(boot_net,
     labels = FALSE,
     order = "sample")
plot(boot_net,"strength",plot = "difference") # 指定输出强度的差异图
plot(boot_net,"edge",plot = "difference") # 指定输出边权重的差异图


case_drop_boot_net <- bootnet(net_bfi,
                              nBoots = 100,
                              nCores = 8,
                              type = "case", # case-drop bootstrap
                              statistics = c('strength',
                                             'expectedInfluence',
                                             'betweenness',
                                             'closeness',"edge"))
plot(case_drop_boot_net, 'all') # 中心性指标随着case-drop率变化的折线图
corStability(case_drop_boot_net) # 中心性指标的稳定性系数


############################# 6. 桥中心性指标  #############################
# net_bfi$graph 提取邻接矩阵（偏相关网络）
bridge_centrality <- bridge(net_bfi$graph, communities = comm_labels)
print(bridge_centrality)
plot(bridge_centrality,order="value",zscore=T) # 绘制桥中心性指标的图
bridge_Strength <- bridge_centrality$`Bridge Strength` # 求桥接节点
# 哪些节点的桥梁强度大于第80百分位数
top_bridges <- names(bridge_Strength[bridge_Strength>quantile(bridge_Strength,probs=0.80,
                                           na.rm=TRUE)])
print(top_bridges)


############################# 7. 桥中心性指标的稳定性检验  #############################
boot_bridge_net <- bootnet(net_bfi, nBoots = 100, nCores = 8,
                           statistics = c("bridgeStrength"),
                           communities = comm_labels # 一定要指定community，否则会报错
)
plot(boot_bridge_net,"bridgeStrength",plot = "difference")
case_boot_bridge_net <- bootnet(net_bfi, nBoots = 100, nCores = 8,
                                statistics = c("bridgeStrength"),
                                communities = comm_labels,
                                type = "case")
plot(case_boot_bridge_net,"bridgeStrength")
corStability(case_boot_bridge_net)