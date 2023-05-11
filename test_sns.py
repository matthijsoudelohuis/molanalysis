import seaborn as sns
import matplotlib.pyplot as plt

dots = sns.load_dataset("dots").query("align == 'dots'")
dots.head()

sns.lineplot(
    data=dots, x="time", y="firing_rate", hue="coherence", style="choice",
)

h = 2

