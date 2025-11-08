from GPC import geo_pattern_causality
import numpy as np
from read_data import read_raster
from trans2M import transtoM


def gpcmain(xMatrix,
    yMatrix,
    E: int = 3,
    tau: int = 1,
    lag: int = 7,
    metric: str = "euclidean",
    weighted: bool = True,
    verbose: bool = True):

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
    
 
    sub_summary1 = {k: summary_y_map_x[k] for k in ["positive", "negative", "dark"]}

    max_key = max(sub_summary1, key=sub_summary1.get)
    print("x->y causality type:", max_key)
    print("x->y causality strenght：", summary_y_map_x["positive"]+summary_y_map_x["negative"]+summary_y_map_x["dark"])

    
    return summary_x_map_y,summary_y_map_x




