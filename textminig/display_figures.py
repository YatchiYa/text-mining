
from modules import *

def diplay_figure_1(dataset):
  figure = plt.figure(figsize=(10, 10))
  targets, sizes = np.unique(dataset.target, return_counts=True)
  target_names = [dataset.target_names[i] for i in targets]
  patches, _, _ = plt.pie(sizes, autopct='%1.1f%%', wedgeprops={'alpha': 0.8})
  plt.legend(patches, target_names, loc=(1, 0.0))
  plt.axis('equal')
  plt.show()


def display_matrix(conf_mat, data):
	# Plot confusion_matrix
	fig, ax = plt.subplots(figsize=(15, 10))
	sns.heatmap(conf_mat, annot=True, cmap = "Set3", fmt ="d", xticklabels=data.target_names, yticklabels=data.target_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show()