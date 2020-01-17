from sklearn.metrics import precision_score

def get_average_precision(y_true, y_pred, labels, average='micro'):
	return precision_score(y_true, y_pred, labels=labels, average=average)
