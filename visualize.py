import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
# 分别设置mathtext公式的正体和斜体字体
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'  # 用于正常数学文本
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 用于斜体数学文本

#plt.rc('text', usetex=True)  # 启用 LaTeX 渲染

def make_real_ratio_diagram():
    # 数据
    x = [5, 10, 25, 50, 75]
    y1 = [ 28.0, 40.9, 56.1, 63.3, 67.3]  # Real Only
    y2 = [32.4, 42.9, 56.9, 63.3, 66.5]  # w/ GLIGEN
    y3 = [39.1, 48.7, 61.0, 67.3, 69.1]  # w/ syn-2
    y3 = [39.5, 49, 61.0, 67.3, 70.2] # w Xsyn-A
    y4 = [37.6, 48.5, 61.1, 66.8, 69.2]  # w/ TIP
    y5 = [36.8, 46.0, 57.8, 63.8, 67.7]  # w/ Xsyn-M

    # 创建图形
    plt.figure(figsize=(7, 4))

    # 绘制曲线
    plt.plot(x, y1, 'g--^', label='Real Only')  # 绿色虚线，三角形标记
    plt.plot(x, y2, 'b--x', label= r'w/ GLIGEN')  # 蓝色虚线，叉号标记
    #plt.plot(x, y4, 'c--s', label='w/ TIP')  # 紫色实线，方形标记
    plt.plot(x, y5, 'y--d', label=r'w/ Xsyn-M')  # 青色实线，菱形标记
    plt.plot(x, y3, 'm--o', label = r'w/ Xsyn-A')  # 橙色实线，圆形标记

    # 从两个点出发画短线
    plt.plot([5, 15], [39.5, 39.5], 'm-', lw=1)  # 短线1
    plt.plot([5, 15], [28.0, 28.0], 'g-', lw=1)  # 短线2
    #plt.plot([5, 15], [37.6, 37.6], 'c-', lw=1)  # 短线3

    y1_point = 39.5
    y2_point = 28.0
    y3_point = 37.6
    x_point = 13
    x_point_2 = 11
    # 在两条短线之间画双向箭头
    plt.annotate('', 
                xy=(x_point, y1_point), 
                xytext=(x_point, y2_point),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5))

    plt.text(x_point+0.5, (y1_point + y2_point) / 2, '+11.5%', color='blue', fontweight='bold', fontsize=13, ha='left')

    # 在两条短线之间画双向箭头
    '''
    plt.annotate('', 
                xy=(x_point_2, y1_point+0.4), 
                xytext=(x_point_2, y3_point-0.2),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5))

    plt.text(x_point_2+0.5, y3_point+0.3, '+1.9%', color='blue', fontweight='bold', fontsize=10, ha='left')
    '''
    # 设置X轴刻度
    plt.xticks(x)

    # 添加图例
    plt.legend(loc='lower right', fontsize=12)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 设置轴标签和标题
    plt.xlabel('The proportion of real data (%)', fontsize=16)
    plt.ylabel('Detection mAP', fontsize=16)

    # 设置刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 显示图形
    plt.tight_layout()
    plt.savefig('real_ratio_w_o_tip_5.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__':

    #make_real_ratio_diagram()
    # 自定义数据 - 请替换为您实际的AP数据
    # 字典键：模型名称，值：包含6个epoch的AP列表
    ap_data = {
        # dkd 70-79
        # 'original': [37.0, 36.6, 35.8, 35.3, 35.4, 34.4, 35.7, 33.3, 35.0, 35.9, 32.4, 36.1],
        # '+unseen': [37.0, 36.2, 36.3, 36.7, 36.1, 36.5, 34.8, 36.1, 36.0, 35.3, 35.7, 37.3],
        # dyq 70-79
        # 'original': [37.4, 37.2, 36.8, 36.5, 36.5, 36.3, 36.0, 35.1, 35.5, 35.0, 34.6, 36.6],
        # '+unseen': [38.4, 37.4, 37.9, 36.8, 35.8, 36.4, 35.8, 36.2, 35.5, 35.3, 35.0, 36.9]
        # dyq 40-79
        'original': [23.2, 20.6, 21.4, 22.4, 19.3, 21.0, 20.4, 20.2, 18.9, 20.2, 21.2, 21.4],
        '+unseen': [24.4, 22.9, 22.7, 20.9, 20.1, 20.5, 21.7, 21.0, 20.4, 20.3, 20.0, 21.4]
        # 'SAGAN': [49.6, 55.4, 61.6, 62.2, 66.2, 69.5],
        # 'Xsyn-A': [50.4, 59.0, 62.9, 65.2, 66.7, 70.7],
        #'CT-GAN': [48.8, 58.2, 62.6, 61.5, 67.5, 69.7],
        #'original': [34.2, 36.5, 36.6, 37.6, 37.8, 37.9, 37.9, 38.7, 38.8, 37.8, 38.6, 41.0],
        
    }

    # 设置图表样式
    plt.figure(figsize=(9, 6), dpi=300)
    # plt.style.use('seaborn-whitegrid')
    # plt.rcParams.update({
    #     'font.family': 'sans-serif',
    #     'axes.edgecolor': '#444444',
    #     'grid.color': '#dddddd'
    # })

    # 定义每个模型对应的颜色
    model_colors = {
        '+unseen': '#FF0000',           # 高饱和红色 (原绿色位置)
        'original': '#0000FF',         # 高饱和蓝色 (原黄色位置)
        # 'CT-GAN': '#FFD700',        # 金黄色 (原天蓝位置)
        # 'Xsyn-A': '#FF0000',        # 高饱和紫色 (原紫色位置) 
    }

    # 绘制每条曲线
    epochs = np.arange(1, 13)  # 从1到6的epoch
    for model, ap_values in ap_data.items():
        plt.plot(
            epochs,
            ap_values,
            color=model_colors[model],
            linewidth=2.5,
            marker='o',
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=1.5,
            label=model
        )

    # 设置坐标轴和标题
    # plt.title('Validation mAP Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=20, labelpad=10)
    plt.ylabel('Average Precision (AP)', fontsize=20, labelpad=10)
    plt.xticks(epochs)  # X轴刻度为1到6
    plt.ylim(18, 25)    # Y轴范围设为20-36，包含所有数据点

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例（放置在右上角）
    plt.legend(
        loc='upper left',
        frameon=True,
        edgecolor='#dddddd',
        fancybox=False,
        shadow=False,
        fontsize=12,
        ncol=1
    )

    # 添加边框
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['top'].set_alpha(0.5)
    plt.gca().spines['right'].set_alpha(0.5)

    # 设置刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 显示图表
    plt.tight_layout(pad=2.0)
    plt.savefig('validation_mAP(dyq 40-79).jpg', dpi=300, bbox_inches='tight')