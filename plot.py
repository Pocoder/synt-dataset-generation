import matplotlib.pyplot as plt
import csv


def plot_loss(file_name1, file_name2 = None):
    epochs = 135
    # first file (train data)
    print('data set 1')

    with open(file_name1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        loss_data1 = list(csv_reader)

    # average epoch
    average_loss1 = []
    for i in range(epochs):
        losses1 = []
        for row in loss_data1[1:]:
            if int(row[0]) == i+1:
                losses1.append(float(row[2]))
        average1 = sum(losses1) / len(losses1)
        average_loss1.append(average1)

    x1 = range(len(average_loss1))
    
    if file_name2 is not None:
        with open(file_name2) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            loss_data2 = list(csv_reader)

        # average epoch
        losses2 = []
        for row in loss_data2[1:]:
            if int(row[0]) == 1:
                losses2.append(float(row[2]))
        average2 = sum(losses2) / len(losses2)
        average_loss2 = [average2] * 120
        x2 = range(len(average_loss2))

    x_label = range(0, epochs, 10)
    y = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.01]

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(x1, average_loss1, label='train')
    if file_name2 is not None:
        plt.plot(x2, average_loss2, label='test')
    plt.legend()

    plt.xticks(x_label, x_label)
    plt.yticks(y, y)
    plt.axis([0, epochs, 0, 0.01])

    plt.xlabel('Эпоха')
    plt.ylabel('Функция потерь')

    plt.savefig('/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_container_002_half_resolution/loss_plot.png')

    plt.show()


plot_loss('/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_container_002_half_resolution/loss_train.csv')
print('done')
