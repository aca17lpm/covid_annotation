from store_retweets import StoreRetweets
import networkx as nx

# apr15misinfo = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020", "covid19misinfo-2020-04", 10000)
# apr15misinfo.classify_misinfo()


# apr15misinfo = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020", "covid19misinfo-2020-04", 10000)
# apr15misinfo.pull_quote_chain()
# apr15misinfo.calculate_retweets()
# apr15misinfo.classify_misinfo()
# nx.write_gexf(apr15misinfo.quoteG, "apr15fullmisinfo.gexf")

# apr15daymisinfo = StoreRetweets("Fri May 15 00:00:00 +0000 2020", "Fri May 15 23:59:59 +0000 2020", "covid19misinfo-2020-05", 10000)
# apr15daymisinfo.pull_quote_chain()
# apr15daymisinfo.calculate_retweets()
# apr15daymisinfo.classify_misinfo()
# nx.write_gexf(apr15daymisinfo.quoteG, "may15daymisinfo.gexf")

# apr15daymisinfo = StoreRetweets("Mon Jun 15 00:00:00 +0000 2020", "Mon Jun 15 23:59:59 +0000 2020", "covid19misinfo-2020-06", 10000)
# apr15daymisinfo.pull_quote_chain()
# apr15daymisinfo.calculate_retweets()
# apr15daymisinfo.classify_misinfo()
# nx.write_gexf(apr15daymisinfo.quoteG, "june15daymisinfo.gexf")

# apr15daymisinfo = StoreRetweets("Wed Jul 15 00:00:00 +0000 2020", "Wed Jul 15 23:59:59 +0000 2020", "covid19misinfo-2020-07", 10000)
# apr15daymisinfo.pull_quote_chain()
# apr15daymisinfo.calculate_retweets()
# apr15daymisinfo.classify_misinfo()
# nx.write_gexf(apr15daymisinfo.quoteG, "july15daymisinfo.gexf")

# apr15daymisinfo = StoreRetweets("Sat Aug 15 00:00:00 +0000 2020", "Sat Aug 15 23:59:59 +0000 2020", "covid19misinfo-2020-08", 10000)
# apr15daymisinfo.pull_quote_chain()
# apr15daymisinfo.calculate_retweets()
# apr15daymisinfo.classify_misinfo()
# nx.write_gexf(apr15daymisinfo.quoteG, "august15daymisinfo.gexf")

# apr15misinfo = StoreRetweets("Wed Apr 15 16:00:00 +0000 2020", "Wed Apr 15 19:00:00 +0000 2020", "covid19misinfo-2020-04", 10000)
# apr15misinfo.pull_quote_chain()
# apr15misinfo.calculate_retweets()
# nx.write_gexf(apr15misinfo.quoteG, "apr15misinfo.gexf")

# apr16misinfo = StoreRetweets("Thu Apr 16 16:00:00 +0000 2020", "Thu Apr 16 19:00:00 +0000 2020", "covid19misinfo-2020-04", 10000)
# apr16misinfo.pull_quote_chain()
# apr16misinfo.calculate_retweets()
# nx.write_gexf(apr16misinfo.quoteG, "apr16misinfo.gexf")

# apr15all = StoreRetweets("Thu Apr 16 17:00:00 +0000 2020", "Thu Apr 16 18:00:00 +0000 2020", "covid19all-2020-04", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "apr15all.gexf")

# apr15all = StoreRetweets("Wed Apr 15 06:00:00 +0000 2020", "Wed Apr 15 08:00:00 +0000 2020", "covid19all-2020-04", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "apr15all6to8.gexf")

# apr15all = StoreRetweets("Wed Apr 15 12:00:00 +0000 2020", "Wed Apr 15 14:00:00 +0000 2020", "covid19all-2020-04", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "apr15all12to2.gexf")

# apr15all = StoreRetweets("Fri May 15 06:00:00 +0000 2020", "Fri May 15 08:00:00 +0000 2020", "covid19all-2020-05", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "may15all6to8.gexf")

# apr15all = StoreRetweets("Fri May 15 12:00:00 +0000 2020", "Fri May 15 14:00:00 +0000 2020", "covid19all-2020-05", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "may15all12to2.gexf")

# apr15all = StoreRetweets("Mon Jun 15 06:00:00 +0000 2020", "Mon Jun 15 08:00:00 +0000 2020", "covid19all-2020-06", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "jun15all6to8.gexf")

# apr15all = StoreRetweets("Mon Jun 15 12:00:00 +0000 2020", "Mon Jun 15 14:00:00 +0000 2020", "covid19all-2020-06", 10000)
# apr15all.pull_quote_chain()
# apr15all.calculate_retweets()
# apr15all.classify_misinfo()
# nx.write_gexf(apr15all.quoteG, "jun15all12to2.gexf")

apr15daymisinfo = StoreRetweets("Tue Sep 15 00:00:00 +0000 2020", "Tue Sep 15 23:59:59 +0000 2020", "covid19misinfo-2020-09", 10000)
apr15daymisinfo.pull_quote_chain()
apr15daymisinfo.calculate_retweets()
apr15daymisinfo.classify_misinfo()
nx.write_gexf(apr15daymisinfo.quoteG, "sep15daymisinfo.gexf")