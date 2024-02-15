# nvBench: Natural Language to Visualization (NL2VIS) Benchmarks

nvBench is a large dataset for complex and cross-domain NL2VIS task, which covers 105 domains, supports seven common types of visualizations, and contains 25,750 (NL, VIS) pairs.
This repository contains the corpus of NL2VIS, with JSON format and Vega-Lite format.

## Introduction to nvBench

- **nvBench.json** stores the JSON format of (NL, VIS) pairs in the nvBench benchmark.

- **nvBench_VegaLite** contains all (NL, VIS) pairs in the nvBench benchmark, and renders the VIS using the [Vega-Lite](https://vega.github.io/vega-lite/) visualization library.

- **database** contains all databases used by the NVBench benchmark.

### nvBench.json

#### (NL, VIS) JSON format
Each (NL, VIS) pair is denoted as a JSON object in NVBench.json, with the following fields:
- `key`: the id of the (NL, VIS) pair in NVBench benchmark
- `vis_query`: contains the query for VIS, with two parts: `vis_part` and `data_part`.
- `chart`: the visualization types: `Bar`, `Pie`, `Line`, `Scatter`, `Stacked Bar`, `Grouping Line`, and `Grouping Scatter`.
- `db_id`: the visualization comes from which database.
- `vis_obj`: the JSON format for representing a visualization object, with `chart` (chart type), `x_name` (name of the X-axis), `y_name`(name of the Y-axis), `x_data` (data for the X-axis), `y_data` (data for the Y-axis), `classify` (Z-axis data, for stacked bar, grouping line, and grouping scatter chart.)
- `nl_queries`: contains the NL queries for querying this visualization object.

Below is an example:
```
"8": {
        "vis_query": {
            "vis_part": "Visualize PIE",
            "data_part": {
                "sql_part": "SELECT Rank , COUNT(Rank) FROM Faculty GROUP BY Rank",
                "binning": ""
            },
            "VQL": "Visualize PIE SELECT Rank , COUNT(Rank) FROM Faculty GROUP BY Rank"
        },
        "chart": "Pie",
        "hardness": "Easy",
        "db_id": "activity_1",
        "vis_obj": {
            "chart": "pie",
            "x_name": "Rank",
            "y_name": "CNT(Rank)",
            "x_data": [
                [
                    "AssocProf",
                    "AsstProf",
                    "Instructor",
                    "Professor"
                ]
            ],
            "y_data": [
                [
                    8,
                    15,
                    8,
                    27
                ]
            ],
            "classify": [],
            "describe": "GROUP BY Rank"
        },
        "nl_queries": [
            "A pie chart showing the number of faculty members for each rank.",
            "What is the number of the faculty members for each rank? Return a pie.",
            "Compute the total the number of rank across rank as a pie chart."
        ]
    }
```

Citation
===========================
When you use the nvBench dataset and the corresponding baseline models, we would appreciate it if you cite the following:

```
@inproceedings{nvBench_SIGMOD21,
  author    = {Yuyu Luo and
               Nan Tang and
               Guoliang Li and
               Chengliang Chai and
               Wenbo Li and
               Xuedi Qin},
  title     = {Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks},
  booktitle = {Proceedings of the 2021 International Conference on Management of
               Data, {SIGMOD} Conference 2021, June 20–25, 2021, Virtual Event, China},
  publisher = {{ACM}},
  year      = {2021},
}
```

NL2VIS Baselines
===========================
Please adapt the Seq2Seq Baselines at the [Spider repository](https://github.com/taoyds/spider/tree/master/baselines/seq2seq_attention_copy). Replace the data preprocessing part and fed the (NL, VIS) pairs of nvBench for training and testing.

Publications
===========================
For more details, please refer to our research paper.
- Yuyu Luo, Nan Tang, Guoliang Li, et al. [Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks](https://luoyuyu.vip/files/nvBench-SIGMOD21.pdf). **SIGMOD 2021**

Contributors
===========================
|#|Contributor|Affiliation|Contact|
|---|----|-----|-----|
|1|[Guoliang Li](http://dbgroup.cs.tsinghua.edu.cn/ligl/)|Professor, Tsinghua University| LastName+FirstName@tsinghua.edu.cn
|2|[Nan Tang](http://da.qcri.org/ntang/index.html)|Senior Scientist, Qatar Computing Research Institute|ntang@hbku.edu.qa
|3|[Yuyu Luo](https://luoyuyu.vip)| PhD Student, Tsinghua University| luoyy18@mails.tsinghua.edu.cn
##### If you have any questions or feedbacks about this project, please feel free to contact Yuyu Luo (luoyy18@mails.tsinghua.edu.cn).


License
===========================
nvBench is available under the
[MIT license](https://opensource.org/licenses/MIT).
