


from matplotlib import pyplot as plt
import seaborn as sns

# HEATMAP ONLY FOR CORRELATION BETWEEN FEATURES AND TARGET (NO FEATURE FEATURE)
def heatmap_against_target(df, features, target_feature):
    corr = df[features].corr()
    target_corr = corr.loc[[target_feature]]

    num_cols = len(features)
    chunk_size = 10
    num_chunks = num_cols // chunk_size + (num_cols % chunk_size > 0)
    fig, axes = plt.subplots(num_chunks, 1, figsize=(15, num_chunks*2.5))

    if num_chunks == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        start = i * chunk_size
        end = start + chunk_size
        subset_corr = target_corr.iloc[:, start:end]

        sns.heatmap(subset_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor="black", ax=ax)

    plt.tight_layout()
    plt.show()

# HEATMAP ONLY FOR FEATURE CORRELATIONS
def heatmap_only_features(df, features):
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor="black")
    plt.show()

# HISTOGRAM FOR ALL FEATURES
def feature_hist(df, features):
    df.hist(grid=False, figsize=(15,15), edgecolor='black', alpha=0.7, bins=15)
    plt.show()


def feature_box_plot(df, features):
    for f in features:
        plt.boxplot(df[f])
        plt.title(f)
        plt.show()
    
