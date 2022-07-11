from store_retweets import StoreRetweets
import networkx as nx

may15daymisinfo = StoreRetweets("Fri May 15 00:00:00 +0000 2020", "Fri May 15 23:59:59 +0000 2020", "covid19misinfo-2020-05", 100)
may15daymisinfo.pull_quote_chain()
may15daymisinfo.calculate_retweets()
may15daymisinfo.classify_misinfo()
nx.write_gexf(may15daymisinfo.quoteG, "test.gexf")
