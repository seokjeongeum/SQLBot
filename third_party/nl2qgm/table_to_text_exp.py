from table2text.model import Table2TextModel


data = [
    {
        "nl_query": "Return the homepage of SIGMOD",
        "table": [{"homepage url": "http://www.sigmod.org/"}],
    },
    {
        "nl_query": "Return the papers whose title contains ’OASSIS’",
        "table": [
            {"paper title": "OASSIS: query driven crowd mining"},
            {"paper title": "OASSIS: Onboard Adaptive Safe-site Identification System"},
        ],
    },
    {
        "nl_query": "Return the papers which were published in conferences in database area",
        "table": [
            {
                "paper title": '"SLAM: Efficient Sweep Line Algorithms for Kernel Density Visualization"'
            },
            {
                "paper title": '"Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale"'
            },
            {
                "paper title": '"Sherman: A Write-Optimized Distributed B+Tree Index on Disaggregated Memory"'
            },
            {"paper title": '"P4DB - The Case for In-Network OLTP"'},
            {
                "paper title": '"HET-GMP: a Graph-based System Approach to Scaling Large Embedding Model Training"'
            },
            {
                "paper title": "Compact Walks: Taming Knowledge-Graph Embeddings with Domain- and Task-Specific Pathways"
            },
            {"paper title": "Rethinking Stateful Stream Processing with RDMA"},
            {"paper title": "Optimizing Recursive Queries with Progam Synthesis"},
            {
                "paper title": "Triton Join: Efficiently Scaling to a Large Join State on GPUs with Fast Interconnects"
            },
            {
                "paper title": "HYPERSONIC: A Hybrid Parallelization Approach for Scalable Complex Event Processing"
            },
        ],
    },
    {
        "nl_query": "Return the authors who published papers in SIGMOD after 2005",
        "table": [
            {"author name": "Tsz Nam Chan"},
            {"author name": "Leong Hou U"},
            {"author name": "Byron Choi"},
            {"author name": "Jianliang Xu"},
            {"author name": "Barrie Kersbergen"},
            {"author name": "Olivier Sprangers"},
            {"author name": "Sebastian Schelter"},
            {"author name": "Qing Wang"},
            {"author name": "Youyou Lu"},
            {"author name": "Jiwu Shu"},
        ],
    },
    {
        "nl_query": "Return the authors who published papers in database conferences",
        "table": [
            {"author name": "Tsz Nam Chan"},
            {"author name": "Leong Hou U"},
            {"author name": "Byron Choi"},
            {"author name": "Jianliang Xu"},
            {"author name": "Barrie Kersbergen"},
            {"author name": "Olivier Sprangers"},
            {"author name": "Sebastian Schelter"},
            {"author name": "Qing Wang"},
            {"author name": "Youyou Lu"},
            {"author name": "Jiwu Shu"},
            {"author name": "Matthias Jasny"},
            {"author name": "Lasse Thostrup"},
        ],
    },
    {
        "nl_query": "Return the organization of authors who published papers in database conferences after 2005",
        "table": [
            {"organization name": "University of Macau"},
            {"organization name": "Hong Kong Baptist University"},
            {"organization name": "University of Amsterdam"},
            {"organization name": "Peking University"},
            {"organization name": "NCSU"},
        ],
    },
]


def main():
    model = Table2TextModel()
    for datum in data:
        result = model.infer(table_in_dict=datum["table"], nl_query=datum["nl_query"])
        print("NL Query: ", datum["nl_query"])
        print("Summary: ", result)


if __name__ == "__main__":
    main()
    pass
