# P.~Mandrik, 2021
# https://github.com/pmandrik/AnimeRecommendationSystem2020
# based on https://www.kaggle.com/hernan4444/anime-recommendation-database-2020

import pandas
from matplotlib import pyplot as plt
from matplotlib import colors
from collections import defaultdict
import numpy as np

def get_colors(N_colors):
  hcolors = [ "416aa3", "64b1af", "8d6268", "c28d75", "387080", "67ad73", "9a4f50", "c6868e", "6e6962", "9d9f7f", "666092", "a593a5",
    "8b5580", "93a167", "99e65f", "5d6872", "7e9e99", "0cf1ff", "f5555d", "3003d9"
   ]
  return [ tuple(int(h[i:i+2], 16)/255. for i in (0, 2, 4)) for h in hcolors  ]

###################################### Explore data from "anime.csv" ######################################
all_data = pandas.read_csv("anime.csv", dtype=str) 
if False : 
  print ( all_data )
  print ("anime.csv feathes = ", list(all_data.columns))

###################################### distribution of number of voters
if False : 
  for x in zip(anime_data['Score-1'], anime_data['Score-2'], anime_data['Score-3'], anime_data['Score-4'], anime_data['Score-5'], anime_data['Score-6'], anime_data['Score-7'], anime_data['Score-8'], anime_data['Score-9'], anime_data['Score-10']):
    def f(x) :
      if x == "Unknown" : return 0
      return float(x)
    if( sum(map(f, x)) > 1000 ) : continue
    print(x, sum(map(f, x)))


################### get Studios
Studios = defaultdict(int)
for val in all_data['Studios']:
  for v in val.split(','):
    Studios[ v.strip() ] += 1

raw = list(dict(Studios).items())
s_sorted = list(reversed(sorted( raw, key=lambda x : x[1])))

if False:
  print(s_sorted[0:50])

################### get scores
Scores = defaultdict(int)
for val in all_data['Score']:
  Scores[ str(val) ] += 1
if False:
  print("anime scores = ", dict(Scores), sum(Scores.values()) )

################### get genres
Genders = defaultdict(int)
for val in all_data['Genders']:
  for v in val.split(','):
    Genders[ v.strip() ] += 1

if False:
  print("anime genders = ", dict(Genders), sum(Genders.values()) )

################### get durations
Durations = defaultdict(int)
for val in all_data['Duration']:
  Durations[ val ] += 1

if False:
  print("anime Durations = ", dict(Durations), sum(Durations.values()) )

################### get episodes
Episodes = defaultdict(int)
for val in all_data['Episodes']:
  if val == "Unknown" : 
    Episodes[ 0 ] += 1;
  else : Episodes[ int(val.strip()) ] += 1

if False:
  print("anime Episodes = ", dict(Episodes), sum(Episodes.values()) )
#for val in sorted(dict(Episodes).keys()):  print(val, Episodes[val] )

################### get Types
Types = defaultdict(int)
for val in all_data['Type']:
  Types[ val.strip() ] += 1

if False:
  print("anime types = ", dict(Types), sum(Types.values()) )

################### get Aired fitches
Month = defaultdict(int)
Year  = defaultdict(int)
years  = []
months = []
years_classes  = []
old_titles = []
for val, name, score in zip(all_data['Aired'], all_data['Name'], all_data['Score']):
  vr = val.split()
  y = 'Unknown'
  m = 'Unknown'
  for v in vr:
    if v.isdigit() and len(v) == 4 :
      y = v
      break
  for v in vr:
    if not v.isdigit() and len(v) >= 3 and v[0].isupper() and v != 'Unknown' :
      m = v[:3]
      break

  Month[ m ] += 1
  Year [ y ] += 1
  years += [ y ]
  months += [ m ]

  if y != 'Unknown' :
    years_classes += [ str(int(y)//5 * 5) + "-" + str(int(y)//5 * 5 + 5) ]
  else : 
    years_classes += [ 'Unknown' ]

  if not y.isdigit() : continue
  if int(y) < 1950 :
    old_titles += [ [y, name, score] ]

if False :
  for t in sorted( old_titles, key=lambda x : x[0] ):
    print( t[1], " (",t[0],",", t[2],")", sep='', end=', ')

if False:
  print("anime Month = ", dict(Month).keys(), sum(Month.values()) )
  print("anime Year = ", dict(Year).keys(), sum(Year.values()) )
all_data['Year'] = years
all_data['Month'] = months
all_data['Year_class'] = years_classes

###################################### Plots ############################################################################
score_var  = 'Score'
fitches_t0 = ['Genders', 'Type', 'Source', 'Rating', 'Year_class', 'Month'] # average + errors
fitches_t1 = ['Members', 'Plan to Watch', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped'] # graph 
fitches_t2 = ['Producers', 'Licensors', 'Studios'] # plot average ???

################### Boxplots/Violinplot + Piecharts ######################################
sort_by_raiting = False
if False : 
  for f in fitches_t0 :
    all_data[f] = all_data[f].str.split(', ')
    
    datas = defaultdict(list)
    for score, types in zip( all_data[score_var], all_data[f] ):
      if score == "Unknown": 
        score = 0
        continue
      for type in types :
        datas[ type ] += [ float(score) ]

    sorted_datas = sorted(datas.items(), key=lambda f : f[0] )
    sorted_datas = [ item for item in reversed(sorted_datas) ]

    if sort_by_raiting : 
      if   f == 'Year_class' : 
        sorted_datas = sorted(datas.items(), key=lambda f : int(f[0].split("-")[0]) if f[0] !='Unknown' else 0 )
      elif f == 'Month'    :    
        sorted_datas = sorted(datas.items(), key=lambda f : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Unknown'].index( f[0] ) )
      else :
        sorted_datas = sorted(datas.items(), key=lambda f : sum(f[1])/len(f[1]) )

    fig, ax = plt.subplots()
    ax.set_title( "Anime " + f )
    plt.gcf().subplots_adjust(left=0.20)
    if f == "Genders" : 
      ax.set_title( "Anime Genre" )
      fig.set_figheight( 2*fig.get_figheight() )
    if f == "Rating" : 
      ax.set_title( "Anime Age rating" )
      plt.gcf().subplots_adjust(left=0.35)
    if f == 'Year_class' : 
      ax.set_title( "Anime Release year" )
    if f == 'Month' : 
      ax.set_title( "Anime Release month" )
    if True  : 
      ax.boxplot( [f[1] for f in sorted_datas], vert=False, flierprops=dict(markerfacecolor='g', marker='D',markersize=2) )
      ax.set_yticklabels( [f[0] for f in sorted_datas] )
    if False  : 
      ax.violinplot( [f[1] for f in sorted_datas], vert=False, showmeans=True, showmedians=False )
      ax.set_yticks( [1+i for i in range(len(sorted_datas))] )
      ax.set_yticklabels( [f[0] for f in sorted_datas] )

    ax.set_xlabel('MyAnimelist Score')
    plt.grid(axis = 'x',  linestyle='-', linewidth=1, color=get_colors(10)[2])
    plt.savefig('images/box_' + f + '.pdf', bbox_inches='tight')

    # pie
    fig, ax = plt.subplots()
    sorted_datas = sorted(datas.items(), key=lambda f : sum(f[1]) )

    data_x = [len(f[1]) for f in sorted_datas]
    data_y = [f[0] for f in sorted_datas]

    if f == 'Source':
      data_x = data_x[-9:] + [ sum(data_x[:-9]) ]
      data_y = data_y[-9:] + [ '...' ]
    if f == 'Genders':
      data_x = data_x[-16:] + [ sum(data_x[:-18]) ]
      data_y = data_y[-16:] + [ '...' ]
    if f == 'Year_class':
      data_x = data_x[-8:] + [ sum(data_x[:-8]) ]
      data_y = data_y[-8:] + [ '...' ]

    patches, texts, autotexts = ax.pie(data_x, labels = data_y, autopct='%d%%')
    ax.set(aspect="equal", title="Fraction of Anime films")
    plt.setp(autotexts, size=12, weight="bold", color="white")

    plt.title("% of Anime per " + f)
    if f == "Genders" : plt.title( "% of Anime per Genre" )
    if f == "Rating"  : plt.title( "% of Anime per Age rating" )
    if f == 'Year_class' : plt.title( "% of Anime per Release year" )
    if f == 'Month' :      plt.title( "% of Anime per Release month" )
    plt.savefig('images/pie_' + f + '.pdf', bbox_inches='tight')
    # plt.show()
  exit()

################### Stackplots per Year ######################################
if False:
  per_4_years = True  

  total_number_of_animas = defaultdict(int)
  for f in ['Genders', 'Type', 'Source', 'Rating'] :
    all_data[f] = all_data[f].str.split(', ')
    datas = {}
    for year in [ str(y) for y in range(1910, 2030) ]:
      datas[ year ] = defaultdict(list)

    all_types = []
    total_number_of_animas = defaultdict(int)
    for score, types, year, air_type in zip( all_data[score_var], all_data[f], all_data["Year"], all_data["Type"]  ):
      if year == "Unknown":  continue
      if score == "Unknown": continue
      if air_type not in ["TV"] : continue
      for type in types :
        if year not in datas : continue
        if per_4_years : 
          yup_min = (int(year)//4 + 0) * 4
          yup_max = (int(year)//4 + 1) * 4
          for y in range(yup_min, yup_max):
            datas[str(y)][ type ] += [ float(score) ]
            total_number_of_animas[str(y)] += 1
        else :
          datas[year][ type ] += [ float(score) ]
          total_number_of_animas[year] += 1
        all_types += [ type ]

    x_data = defaultdict(list)
    for type in list(set(all_types)):
      for year in [ str(y) for y in range(1958, 2021) ]:
        x_data[ type ] += [ len( datas[ year ][ type ] ) ]

    sorted_datas = sorted(x_data.items(), key=lambda f : -sum(f[1]) )

    for i in range(len(sorted_datas[0][1])) :
      summ = sum( [ ff[1][i] for ff in sorted_datas ] )
      if summ == 0 : continue
      for j in range(len(sorted_datas)):
        sorted_datas[j][1][i] /= summ

    if len(sorted_datas) > 18:
      others = []
      for i in range(len(sorted_datas[0][1])) :
        summ = sum( [ ff[1][i] for ff in sorted_datas[18:] ] )
        others += [ summ ]
      sorted_datas = sorted_datas[:18] + [ ["...", others] ]

    colorz = get_colors( 100 )

    fig, ax = plt.subplots()
    ax.stackplot([y for y in range(1958, 2021)], [f[1] for f in sorted_datas], labels=[f[0] for f in sorted_datas], colors=colorz )
    ax.set_xlabel('Year')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.subplots_adjust(right=0.5)
    fig.set_figwidth( 2*fig.get_figwidth() )

    ax.set_title( "Anime " + f )
    if f == "Genders" :    ax.set_title( "Anime Genre" )
    if f == "Rating" :     ax.set_title( "Anime Age rating" )
    if f == 'Year_class' : ax.set_title( "Anime Release year" )
    if f == 'Month' :      ax.set_title( "Anime Release month" )

    ax.set_ylabel('Fraction of Anime')
    plt.savefig('images/stack_' + f + '.pdf', bbox_inches='tight')

  if total_number_of_animas:
    total_number_of_animas = sorted(total_number_of_animas.items(), key=lambda f : int(f[0]) )
    print( total_number_of_animas )

    plt.plot( [int(f[0]) for f in total_number_of_animas], [f[1] for f in total_number_of_animas] )
    plt.legend('',frameon=False)
    plt.xlabel('Year')
    plt.title( "Number of released Anime per year" )
    plt.ylabel('Number of Anime')
    plt.savefig('images/stack_total.pdf', bbox_inches='tight')

    cumulative = [ total_number_of_animas[0][1] ]
    for f in  total_number_of_animas[1:]:
      cumulative.append( cumulative[-1] + f[1] )
      print( cumulative[-1], f[1] )
    plt.plot( [int(f[0]) for f in total_number_of_animas], cumulative )
    plt.xlabel('Year')
    plt.title( "Total number of Anime" )
    plt.ylabel('Number of Anime')
    plt.savefig('images/stack_total_sum.pdf', bbox_inches='tight')
  exit()

################### Number of Voters ######################################
if False:
  # Add number of voters
  def get_voters(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    def f(val) :
      if val == "Unknown" : return 0
      return float(val)
    return int(sum(map(f, x)))
  all_data["N_votes"] = all_data.apply(lambda x: get_voters(x['Score-1'], x['Score-2'], x['Score-3'], x['Score-4'], x['Score-5'], x['Score-6'], x['Score-7'], x['Score-8'], x['Score-9'], x['Score-10']), axis=1) 

  for f in fitches_t0 : 
    all_data[f] = all_data[f].str.split(', ')
    datas = defaultdict(list)
    for score, types in zip( all_data["N_votes"], all_data[f] ):
      if score == "Unknown": continue
      for type in types :
        datas[ type ] += [ float(score) ]

    sorted_datas = sorted(datas.items(), key=lambda f : sum(f[1]) )
    data_x = [sum(f[1]) for f in sorted_datas]
    data_xx = [len(f[1]) for f in sorted_datas]
    data_y = [f[0] for f in sorted_datas]

    for x,y,xx in zip(data_x,data_y,data_xx):
      print(x,y,xx)

    if f == 'Source':
      data_x = data_x[-9:] + [ sum(data_x[:-9]) ]
      data_y = data_y[-9:] + [ '...' ]
    if f == 'Genders':
      data_x = data_x[-16:] + [ sum(data_x[:-18]) ]
      data_y = data_y[-16:] + [ '...' ]
    if f == 'Year_class':
      data_x = data_x[-8:] + [ sum(data_x[:-8]) ]
      data_y = data_y[-8:] + [ '...' ]

    print(data_x, data_y)

    data_x = list(reversed(data_x))
    data_y = list(reversed(data_y))

    fig, ax = plt.subplots()
    data_y = data_y[::2] + list(data_y[1::2])
    wedges, texts, autotexts = ax.pie(data_x[::2] + list(data_x[1::2]), wedgeprops=dict(width=0.5), startangle=-40, autopct='%d%%', pctdistance=0.80)

    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
      ang = (p.theta2 - p.theta1)/2. + p.theta1
      y = np.sin(np.deg2rad(ang))
      x = np.cos(np.deg2rad(ang))
      horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
      connectionstyle = "angle,angleA=0,angleB={}".format(ang)
      kw["arrowprops"].update({"connectionstyle": connectionstyle})
      if data_y[i] == "Unknown" : continue
      if data_y[i] == "Music" : continue
      extra = 0
      if data_y[i] == "Rx - Hentai" : extra = 0.1
      if data_y[i] == "OVA" : extra = 0.1
      ax.annotate(data_y[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.1*y+extra), horizontalalignment=horizontalalignment, **kw)
  
    plt.setp(autotexts, size=10, weight="bold", color="black")
    ax.set_aspect(aspect=1.25)

    plt.title("% of User Votes per Anime " + f)
    if f == "Genders" : plt.title( "% of User Votes per Anime Genre" )
    if f == "Rating"  : plt.title( "% of User Votes Anime Age rating" )
    if f == 'Year_class' : plt.title( "% of User Votes per Anime Release year" )
    if f == 'Month' :      plt.title( "% of User Votes per Anime Release month" )
    plt.savefig('images/pie_nvotes_' + f + '.pdf', bbox_inches='tight')
  exit()

################### Create groups of animas base on number of episodes
if False:
  N_episodes_regions = [1, 5, 10, 20, 30, 45, 70, 100, 9000]
  # number of 'Episodes'
  for N in N_episodes_regions :
    all_data[ 'Episodes_' + str(N) ] = 0
    
  episodes = []
  for i in all_data.index:
    eps = all_data.at[i, 'Episodes']
    if eps == "Unknown" : continue
    eps = int(eps)
    for N in N_episodes_regions :
      if eps > N : continue
      all_data.at[i, 'Episodes_' + str(N) ] = 1
      break
    if eps < 2 : continue
    if eps > 100 : continue
    episodes += [ eps ]

  fig, ax = plt.subplots(tight_layout=True)
  hist = ax.hist(episodes, bins=50, facecolor='g')
  ax.set_xlabel('Number of series (>1)')
  ax.set_ylabel('Number of Anime')
  # plt.yscale('log')
  plt.savefig('images/hist_N_episodes.pdf', bbox_inches='tight')

  datas = defaultdict(list)
  for i, N in enumerate(N_episodes_regions) :
    title  = 'Episodes_' + str(N)
    xtitle = str( ([0] + N_episodes_regions)[i] ) + "-" + str(N_episodes_regions[i])
    datas[ xtitle ] = all_data[all_data[title] == 1][all_data["Score"] != "Unknown"]["Score"].tolist()

  sorted_datas = sorted(datas.items(), key=lambda f : int(f[0].split("-")[0]) )

  fig, ax = plt.subplots()
  ax.set_title( "Anime Number of episodes" )
  ax.boxplot( [[float(x) for x in f[1]] for f in sorted_datas], vert=False, flierprops=dict(markerfacecolor='g', marker='D') )
  ax.set_yticklabels( [f[0] for f in sorted_datas] )
  ax.set_xlabel('MyAnimelist Score')
  plt.savefig('images/box_nepisodes.pdf', bbox_inches='tight')

  # pie
  fig, ax = plt.subplots()

  data_x = [len(f[1]) for f in sorted_datas]
  data_y = [f[0] for f in sorted_datas]

  patches, texts, autotexts = ax.pie(data_x, labels = data_y, autopct='%d%%')
  ax.set(aspect="equal", title="Fraction of Anime films")
  plt.setp(autotexts, size=12, weight="bold", color="white")

  plt.title("% of Anime per Number of episodes")
  plt.savefig('images/pie_nepisodes.pdf', bbox_inches='tight')
  exit()

################### create groups of animas base on duration
if False :
  print("Duration ... ")
  times = []
  weights = []
  def get_time( time_raw ):
    items = time_raw.split()
    time = -1
    for i, val in enumerate(items):
      if val == "sec." : 
        time += int( items[i-1] )/60.
      if val == "min." : 
        time += int( items[i-1] )
      if val == "hr." : 
        time += int( items[i-1] )*60.
    return time

  for key, value in Durations.items():
    times += [ get_time( key ) ]
    weights += [ value ]

  fig, ax = plt.subplots(tight_layout=True)
  hist = ax.hist(times, weights=weights, bins= np.arange(0, 175, 175/50.) , facecolor='g', edgecolor='#169acf',)
  ax.set_xlabel('Duration of Anime per episode [min.]')
  ax.set_ylabel('Number of Anime')
  plt.savefig('images/hist_duration.pdf', bbox_inches='tight')

  h2_x, h2_y = [], []
  h2_x_year = []
  for key, value in Durations.items():
    time = get_time( key )

    tmp = all_data[all_data['Duration'] == key]
    tmp = tmp[tmp["Score"] != "Unknown"]
    scores = tmp["Score"].to_list()
    h2_x += [ float(s) for s in scores ]
    h2_y += [ time for s in scores ]

    year   = tmp["Year"].to_list()
    h2_x_year += [ float(s) if s != 'Unknown' else 2000 for s in year ]

  fig, ax = plt.subplots(tight_layout=True)
  hist = ax.hist2d(h2_x, h2_y, bins=[np.arange(1, 10, 0.25), np.arange(0, 160, 2.5)], norm=colors.LogNorm())
  ax.set_xlabel('MyAnimelist Score')
  ax.set_ylabel('Duration of Anime per episode [min.]')
  plt.savefig('images/hist2d_duration.pdf', bbox_inches='tight')

  fig, ax = plt.subplots(tight_layout=True)
  hist = ax.hist2d(h2_x_year, h2_y, bins=100, norm=colors.LogNorm())
  ax.set_xlabel('Release Year')
  ax.set_ylabel('Duration of Anime per episode [min.]')
  plt.savefig('images/hist2d_duration_year.pdf', bbox_inches='tight')
    
  exit()

### distribution of duration based on genre
if False :
  def get_time( time_raw ):
    items = time_raw.split()
    time = -1
    for i, val in enumerate(items):
      if val == "sec." : 
        time += int( items[i-1] )/60.
      if val == "min." : 
        time += int( items[i-1] )
      if val == "hr." : 
        time += int( items[i-1] )*60.
    return time

  h1 = []
  h2 = []
  for time_, genre in zip(all_data["Duration"],all_data["Genders"]):
    time = get_time( time_ )

    if "Dementia" in genre : h1 += [ time ]
    else                   : h2 += [ time ]

  fig, ax = plt.subplots(tight_layout=True)
  # hist = ax.hist( [h1, h2], bins=20, histtype='step', linewidth=2, alpha=0.7, label=['Dementia','Not Dementia'])
  plt.hist( h1, bins=50, histtype='step', linewidth=2, alpha=0.7, label=['Dementia'], normed=True)
  plt.hist( h2, bins=50, histtype='step', linewidth=2, alpha=0.7, label=['Not Dementia'], normed= True)
  plt.legend(loc='upper right')

  ax.set_ylabel('Fraction of Anime')
  ax.set_xlabel('Duration of Anime per episode [min.]')
  plt.savefig('images/hist_dem_distribution.pdf', bbox_inches='tight')
    
  exit()


################### 2d hists
if False :
  for f in fitches_t1 :
    x,y = [],[]
    for score, fs in zip( all_data[score_var], all_data[f] ):
      if score == "Unknown": continue
      if fs == "Unknown": continue
      x += [ float(score) ]
      y += [ float(fs) ]
    fig, ax = plt.subplots(tight_layout=True)
    #hist = ax.hist2d(x, y, bins=500, norm=colors.LogNorm())
    hist = ax.plot(x, y, '.', markersize=1.)
    ax.set_xlabel('MyAnimelist Score')
    ax.set_ylabel(f)
    plt.yscale('log')
    plt.savefig('images/hist2d_user_status_vs_' + "".join(f.split()) + '.png', bbox_inches='tight')
  exit()

################### funny correlations
if False :
  all_data[ 'Japanese name - Lenght' ] = all_data.apply(lambda x: len(x['Japanese name']), axis=1)
  all_data[ 'Japanese name - Number of Words' ] = all_data.apply(lambda x: len(x['Japanese name'].split()), axis=1)
  all_data[ 'English name - Lenght' ] = all_data.apply(lambda x: len(x['English name']), axis=1)
  all_data[ 'English name - Number of Words' ] = all_data.apply(lambda x: len(x['English name'].split()), axis=1)

  if False:
    for f in ['Japanese name - Lenght', 'Japanese name - Number of Words', 'English name - Lenght', 'English name - Number of Words'] :
      print(f, " ... TOP 10",)
      data = all_data.sort_values(by=[f])[['Japanese name', 'English name', f]]
      sdata = data.tail(10)
      for a,b,c in zip( sdata['Japanese name'], sdata['English name'], sdata[f] ):
        print(a,b,c)

      print(f, " ... LAST 10",)
      sdata = data.head(10)
      for a,b,c in zip( sdata['Japanese name'], sdata['English name'], sdata[f] ):
        print(a,b,c)

    for f in ['Japanese name - Lenght', 'Japanese name - Number of Words', 'English name - Lenght', 'English name - Number of Words'] :
      fig, ax = plt.subplots(tight_layout=True)

      data = all_data[ all_data['Japanese name'] != 'Unknown' ][ f ].to_list()
      if 'Japanese name' not in f : 
        data = all_data[ all_data['English name'] != 'Unknown' ][ f ].to_list()
      hist = ax.hist( data, facecolor="#64b1af", edgecolor="#416aa3", bins=range(min(data), max(data) + 1, 1) )
      ax.set_xlabel( f, fontsize=18 )
      ax.set_ylabel('Number of Anime', fontsize=18)
      # plt.yscale('log')
      postfix = "".join(f.split())
      postfix = "".join(postfix.split("-"))
      plt.savefig('images/hist_N_' + postfix + '.pdf', bbox_inches='tight',)

  fs = ['Genders', 'Type', 'Source', 'Rating', 'English name - Lenght', 'English name - Number of Words', 'Japanese name - Lenght']
  fs = ['Genders']

  for f in ['Genders', 'Type', 'Source', 'Rating'] : 
    all_data[f] = all_data[f].str.split(', ')

  all_data = all_data[ all_data['Japanese name'] != 'Unknown' ]
  all_data = all_data[ all_data['English name'] != 'Unknown' ]
  for i, fx in enumerate(fs) :
    for j, fy in enumerate(fs) :
      points = []
      if i >= j and not (fx == 'Genders' and fy == 'Genders'): continue
      tmp = all_data[ all_data[fx] != "Unknown" ]
      tmp = tmp[ tmp[fy] != "Unknown" ]

      xx, yy = [], []
      ws = []

      types_vx = []
      types_vy = []
      for x, y, score in zip( tmp[fx], tmp[fy], tmp['Score'] ):
        if not type(x) == type([]): x = [x]
        if not type(y) == type([]): 
          y = [y]
          if not types_vy : 
            types_vy = sorted(tmp[fy].unique())
        for vx in x:
          if fx != "Score" :
            if vx not in types_vx : types_vx += [ vx ]
            vx = types_vx.index(vx)
          else : vx = float( vx )
          for vy in y :
            if vy not in types_vy : types_vy += [ vy ]
            vy = types_vy.index(vy)
            if vy == vx : continue
            xx += [vx]
            yy += [vy]
            if score != "Unknown": ws += [ float(score) ]
            else : ws += [ 0. ]

      print(xx[:10], yy[:10])
      fig, ax = plt.subplots(tight_layout=True)
      if True : # average score
        bin_entries = defaultdict(int)
        bin_weights = defaultdict(int)
        for x,y,w in zip(xx,yy,ws):
          bin_entries[ (x,y) ] += 1 
          bin_weights[ (x,y) ] += w

        ws = []
        xxx, yyy = [], []
        for x in range(int(min(xx)), int(max(xx)) + 2, 1) : 
          for y in range(int(min(yy)), int(max(yy)) + 2, 1) :
            xxx += [x]
            yyy += [y]
            if bin_entries[ (x,y) ] > 0:
              ws  += [ bin_weights[ (x,y) ] /  bin_entries[ (x,y) ] / 10. ]
            else : ws += [ 0 ]

        hist = ax.hist2d(xxx, yyy, weights=ws, bins=[range(int(min(xx)), int(max(xx)) + 2, 1), range(int(min(yy)), int(max(yy)) + 2, 1)], cmap=plt.cm.Oranges)
        fig.colorbar(hist[3], ax=ax)

      else : # number of entries
        hist = ax.hist2d(xx, yy, bins=[range(int(min(xx)), int(max(xx)) + 2, 1), range(int(min(yy)), int(max(yy)) + 2, 1)], norm=colors.LogNorm(), cmap=plt.cm.Reds)
      ax.set_xlabel(".")
      ax.set_ylabel(".")
      ax.xaxis.set_ticklabels([])
      ax.yaxis.set_ticklabels([])

      for count, x in zip(types_vx, range(int(min(xx)), int(max(xx)) + 1, 1) ):
          ax.annotate(str(count), xy=(1*x+0.5, 0), xycoords=('data', 'axes fraction'), xytext=(0, -18), textcoords='offset points', va='top', ha='center', rotation=90)

      for count, x in zip(types_vy, range(int(min(yy)), int(max(yy)) + 1, 1) ):
          ax.annotate(str(count), xy=(0, 1*x+0.5), xycoords=('axes fraction', 'data'), xytext=(-18, 0), textcoords='offset points', va='center', ha='right')

      postfix = "".join((fx + "vs" + fy).split())
      postfix = "".join(postfix.split("-"))

      title = fx + " vs " + fy 

      title = title.replace('Genders', "Genre")
      title = title.replace('Rating', "Age rating")

      ax.set_title( title )
      # plt.colorbar(ax.get_children()[2], ax=ax)
      if (fx == 'Genders' and fy == 'Genders') : fig.set_figheight( 1.25*fig.get_figheight() )
      plt.savefig('images/hist2d_corr_' + postfix + '.pdf', bbox_inches='tight',)
       
  exit()

################### studios and producers
if False:
  for f in fitches_t2 :
    all_data[f] = all_data[f].str.split(', ')
    
    datas = defaultdict(list)
    for score, types in zip( all_data[score_var], all_data[f] ):
      if score == "Unknown": continue
      for type in types :
        if type == "Unknown" : continue
        datas[ type ] += [ float(score) ]

    items = list(datas.items())
    items = sorted(items, key=lambda x : len(x[1]))
    print(f, "total", len(items))
    for ff in reversed(items[-10:]):    
      print( f, ff[0], len( ff[1] ) )

    x = [ sum(f[1])/len(f[1]) for f in items ]
    y = [ len(f[1]) for f in items ]

    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.plot(x, y, '*', markersize=4., color="#659EC7")
    # hist = ax.hist2d(x, y, bins=25, norm=colors.LogNorm())
    ax.set_xlabel('Average MyAnimelist Score',fontsize=15)
    ax.set_ylabel('Number of '  + f + ' Anime',fontsize=15)
    plt.yscale('log')
    plt.savefig('images/hist2d_N_' + f + '.png', bbox_inches='tight',)
  exit()

if False:
  for f in fitches_t2 :
    all_data[f] = all_data[f].str.split(', ')

    animas_per_studio = defaultdict(lambda:0)
    studio_per_anima  = []
    
    datas_l = defaultdict(lambda:0)
    datas_f = defaultdict(lambda:9999)
    for name, year, types in zip( all_data["Name"], all_data["Year"], all_data[f] ):
      if year == "Unknown": continue
      for type in types :
        if type == "Unknown" : continue
        datas_f[ type ] = min(datas_f[ type ], int(year))
        datas_l[ type ] = max(datas_l[ type ], int(year))
        animas_per_studio[ type ] += 1
      if len(types) > 10 :
        print( name, f, types )

      studio_per_anima += [ len([ type for type in types if type != "Unknown" ]) ]

    xf = [ f[1] for f in datas_f.items() ]
    xl = [ f[1] for f in datas_l.items() ]
    delta = [ fl[1] - ff[1] for fl, ff in zip(datas_l.items(), datas_f.items()) ]

    delta_old = []
    for fl, ff in zip(datas_l.items(), datas_f.items()):
      if fl[1] > 2015 : continue 
      delta_old += [ fl[1] - ff[1] ]

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(xf, facecolor="#64b1af", edgecolor="#416aa3", bins=np.arange(min(xf), max(xf), 1))
    ax.set_xlabel('Year of first Anime',fontsize=15)
    ax.set_ylabel('Number of '  + f,fontsize=15)
    plt.savefig('images/hist_N_first' + f + '.pdf', bbox_inches='tight',)

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(xl, facecolor="#64b1af", edgecolor="#416aa3", bins=np.arange(min(xl), max(xl), 1))
    ax.set_xlabel('Year of last Anime',fontsize=15)
    ax.set_ylabel('Number of '  + f,fontsize=15)
    plt.savefig('images/hist_N_last' + f + '.pdf', bbox_inches='tight',)

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(delta, facecolor="#64b1af", edgecolor="#416aa3", bins=np.arange(min(delta), max(delta), 1))
    ax.set_xlabel('Year of last Anime  - Year of first Anime',fontsize=12)
    ax.set_ylabel('Number of '  + f,fontsize=15)
    plt.savefig('images/hist_N_delta' + f + '.pdf', bbox_inches='tight',)

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(delta_old, facecolor="#64b1af", edgecolor="#416aa3", bins=np.arange(min(delta), max(delta), 1))
    ax.set_xlabel('Year of last Anime ( < 2015 )  - Year of first Anime',fontsize=12)
    ax.set_ylabel('Number of '  + f,fontsize=15)
    plt.savefig('images/hist_N_deltaOld' + f + '.pdf', bbox_inches='tight',)

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist( list(animas_per_studio.values()), facecolor="#64b1af", edgecolor="#416aa3", bins=100)
    ax.set_xlabel('Number of Anime',fontsize=15)
    ax.set_ylabel('Number of '  + f,fontsize=15)
    plt.yscale('log')
    plt.savefig('images/hist_N_animas_per_' + f + '.pdf', bbox_inches='tight',)

    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(studio_per_anima, facecolor="#64b1af", edgecolor="#416aa3", bins=np.arange(min(studio_per_anima), max(studio_per_anima), 1))
    ax.set_xlabel('Number of associated '+ f,fontsize=15)
    ax.set_ylabel('Number of Anime',fontsize=15)
    plt.yscale('log')
    plt.savefig('images/hist_N_' + f + '_per_anima.pdf', bbox_inches='tight',)
    
  exit()

################### simple anime score distributions
if False:
  def get_voters(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    def f(val) :
      if val == "Unknown" : return 0
      return float(val)
    return int(sum(map(f, x)))
  all_data["N_votes"] = all_data.apply(lambda x: get_voters(x['Score-1'], x['Score-2'], x['Score-3'], x['Score-4'], x['Score-5'], x['Score-6'], x['Score-7'], x['Score-8'], x['Score-9'], x['Score-10']), axis=1) 

  tmp = all_data[all_data["Score"] != "Unknown"]

  xx = tmp["N_votes"].to_list()
  fig, ax = plt.subplots(tight_layout=True)
  ax.hist(xx, facecolor="#64b1af", edgecolor="#416aa3", bins=50)
  ax.set_xlabel('Number of Votes',fontsize=15)
  ax.set_ylabel('Number of Anime',fontsize=15)
  plt.yscale('log')
  plt.savefig('images/hist_votes.pdf', bbox_inches='tight',)

  yy = [ float(f) for f in tmp["Score"].to_list() ]
  fig, ax = plt.subplots(tight_layout=True)
  ax.hist(yy, facecolor="#64b1af", edgecolor="#416aa3", bins=50)
  ax.set_xlabel('Average Score',fontsize=15)
  ax.set_ylabel('Number of Anime',fontsize=15)
  plt.savefig('images/hist_scores.pdf', bbox_inches='tight',)

  tmp_1 = tmp.sort_values(["N_votes"], ascending=False)
  tmp_1 = tmp_1.head(10)[ ["Name", "N_votes", "Score"] ]
  for a,b,c in zip( tmp_1["Name"], tmp_1["Score"], tmp_1["N_votes"] ) : 
    print( a, "&" , b, "&" , c, "\\\\" )

  tmp_2 = tmp.sort_values(["Score"], ascending=False)
  tmp_2 = tmp_2.head(10)[ ["Name", "N_votes", "Score"] ]
  for a,b,c in zip( tmp_2["Name"], tmp_2["Score"], tmp_2["N_votes"] ) : 
    print( a, "&" , b, "&" , c, "\\\\" )

  exit()

###################################### Top tiers data ### stackplots ### 
if False:
  total_number_of_animas = defaultdict(int)
  tmp = all_data[all_data["Score"] != "Unknown"]
  tmp = tmp.sort_values(['Score'], ascending=False)

  tops = [1] + [10 + x * 10 for x in range(0, 100)]
  reqs = max(tops)

  for f in ['Genders', 'Type', 'Source', 'Rating', 'Year_class'] :
    tmp[f] = tmp[f].str.split(', ')
    all_data_fs = []
    for s, types, name in zip( tmp['Score'], tmp[f], tmp['Name'] ):
      if "Rx - Hentai" in types : 
        print( s, name )
      all_data_fs += [ [types] ]

    all_types = []
    all_results = []
    for limit in tops:
      limit_results = defaultdict(int)
      for types in all_data_fs[:limit] :
        for t in types[0] : 
          limit_results[t] += 1
          if t not in all_types : all_types += [ t ]
      all_results += [ limit_results ]
      print( limit_results )
      print( all_results )

    all_types_sorted = sorted( all_types )
    for i in range(len(all_results)):
      keys = all_results[i].keys()
      for t in all_types :
        if t not in keys : all_results[i][t] = 0

      vals = all_results[i].items()
      vals = sorted( vals, key=lambda x : x[0] )
      summ = sum( [ x[1] for x in vals ] )
      all_results[i] = [ float(x[1]) / summ for x in vals ]

    colorz = get_colors( 100 )

    fig, ax = plt.subplots()
    ax.stackplot( [int(x) for x in tops], [ [x[i] for x in all_results] for i in range(len(all_results[0]))], labels=all_types_sorted, colors=colorz )
    ax.set_xlabel('Top N Anime')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.subplots_adjust(right=0.5)
    fig.set_figwidth( 2*fig.get_figwidth() )

    ax.set_title( "Anime " + f )
    if f == "Genders" :    ax.set_title( "Anime Genre" )
    if f == "Rating" :     ax.set_title( "Anime Age rating" )
    if f == 'Year_class' : ax.set_title( "Anime Release year" )

    ax.set_ylabel('Fraction of Anime')
    plt.savefig('images/stack_top' + f + '.pdf', bbox_inches='tight')
  plt.show()
  exit()

print(__file__, " - please activate corresponding parts of the code to produce plots or print out information!")





