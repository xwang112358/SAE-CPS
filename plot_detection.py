
import matplotlib.pyplot as plt


def plot_detection_trajectory(predicted_labels, y_test):
    """
    Plot detection trajectory comparing predicted and actual attack states.
    
    Args:
        predicted_labels: Array of predicted binary labels (0: no attack, 1: attack)
        y_test: Array of true binary labels (0: no attack, 1: attack)
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    shade_of_gray = '0.75'
    ax.plot(predicted_labels, color=shade_of_gray, label='predicted state')
    ax.fill_between(range(len(predicted_labels)), predicted_labels,
                    where=predicted_labels <= 1, interpolate=True, color=shade_of_gray)

    # Plot real state  
    ax.plot(y_test, color='r', alpha=0.85, lw=5, label='real state')

    # Customize the plot
    ax.set_title('Detection trajectory on test dataset', fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['NO ATTACK', 'ATTACK'])
    ax.legend(fontsize=12, loc=2)

    # Save the plot
    plt.savefig('detection_trajectory.png')
    plt.close()