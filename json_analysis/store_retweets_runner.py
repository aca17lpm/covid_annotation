from store_retweets import StoreRetweets
import networkx as nx

apr15 = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020")

apr15.pull_and_store_tweets(10000)

nx.write_gexf(apr15.G, "test.gexf")