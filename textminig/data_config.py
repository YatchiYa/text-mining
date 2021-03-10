from modules import *

def loading_data(mode):
  dataset_20 = fetch_20newsgroups(subset=mode, shuffle=True, remove=('headers', 'footers', 'quotes'))
  return (dataset_20)

def size_and_names_dataset(dataset):
  len_data = len(dataset['target'])
  target_names = dataset.target_names
  return (len_data, target_names)

# Finding frequency of each category
def finding_frequency_each_data(dataset):
  targets, frequency = np.unique(dataset.target, return_counts=True)
  return (targets, frequency)