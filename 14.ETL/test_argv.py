#!/Users/hkromer/anaconda3/bin/python3

import getopt
import sys
import datetime

if __name__ == '__main__':
   # Get the arguments from the command-line except the filename
   argv = sys.argv[1:]

   try:
      if len(argv) == 3:
         # day = argv[0]
         t_plot_start = argv[0]
         t_plot_end = argv[1]
         output_path = argv[2]
         print(datetime.datetime.now().date())
      else:
         print('usage: 001.ETL.from_live_table.py.py t_plot_start t_plot_end output_path')

   except getopt.GetoptError:
       # Print something useful
       print('usage: 001.ETL.from_live_table.py.py t_plot_start t_plot_end output_path')
       sys.exit(2)