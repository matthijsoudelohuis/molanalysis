import seaborn as sns


dots = sns.load_dataset("dots").query("align == 'dots'")
dots.head()

