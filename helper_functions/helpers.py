import pickle
import sys
import matplotlib.pyplot as plt

def save_pickle(file_path_without_extension, data, feedback=True):
    """
    Save data to a pickle file
    :param file_path_without_extension: path of save location with .pkl omitted
    :param data: data to be saved
    """
    with open(file_path_without_extension + '.pkl', 'wb') as f:
        pickle.dump(data, f)

    if feedback:
        print('Done pickling %s' % file_path_without_extension)


def load_pickle(file_path, feedback=False):
    """
    Load data from pickle file
    :param file_path: path to .pkl file
    :param feedback: if True print messages indicating when loading starts and finishes
    :return: data from pickle file
    """
    if feedback:
        print('Loading %s', file_path)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if feedback:
        print('Done loading pickle')

    return data



# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    source: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = 'X' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_training_history(training_history):
    """

    :param training_history:

    :type training_history: dict
    :return:
    """
    # print(training_history)

    metrics = list(training_history.keys())
    metric_history = list(training_history.values())
    plt.figure(figsize=(40, 40))

    for i in range(len(metrics)):
        plt.plot(metric_history[i], label=metrics[i])

    plt.ylabel('Metric value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    print