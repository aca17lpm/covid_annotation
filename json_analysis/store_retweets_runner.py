from store_retweets import StoreRetweets
import networkx as nx

apr15misinfo = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020", "covid19misinfo-2020-04", 10000)
apr15misinfo.pull_quote_chain()
apr15misinfo.calculate_retweets()
nx.write_gexf(apr15misinfo.quoteG, "apr15misinfo.gexf")

# apr15all = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020", "covid19all-2020-04", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# nx.write_gexf(apr15all.quoteG, "apr15all.gexf")