from GPC_op import geo_pattern_causality
import numpy as np


def gpcmain(xMatrix,
    yMatrix,
    E: int = 3,
    tau: int = 1,
    lag: int = 7,
    metric: str = "euclidean",
    weighted: bool = True,
    verbose: bool = True):
    # 运行分析
    results_x_map_y = geo_pattern_causality(
        xMatrix,
        yMatrix,
        E,
        tau,
        lag,
        metric=metric,
        weighted=weighted,
        verbose=verbose
    )

    summary_x_map_y=results_x_map_y['summary']
    
    
    sub_summary2 = {k: summary_x_map_y[k] for k in ["positive", "negative", "dark"]}
    max_key = max(sub_summary2, key=sub_summary2.get)
    
    print("Parameter：")
    print("E=:",E)
    print("metric=:",metric)
    
    print("y->x causality type:", max_key)
    print("y->x causality strenght：", summary_x_map_y["positive"]+summary_x_map_y["negative"]+summary_x_map_y["dark"])
    
    

        # 运行分析
    results_y_map_x = geo_pattern_causality(
        yMatrix,
        xMatrix,
        E,
        tau,
        lag,
        metric=metric,
        weighted=weighted,
        verbose=verbose
    )
    summary_y_map_x=results_y_map_x['summary']
    
    # 查看输出
    

    # 只取出这三个指标
    sub_summary1 = {k: summary_y_map_x[k] for k in ["positive", "negative", "dark"]}
    # 找出最大值对应的索引（键）
    max_key = max(sub_summary1, key=sub_summary1.get)
    print("x->y causality type:", max_key)
    print("x->y causality strenght：", summary_y_map_x["positive"]+summary_y_map_x["negative"]+summary_y_map_x["dark"])

    
    return summary_x_map_y,summary_y_map_x




