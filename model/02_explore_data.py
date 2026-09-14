import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explore_data(data_dir):
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Could not find Train or Test directories in {data_dir}")
        print("Please run 01_download_data.py first or point to the correct directory.")
        return

    def get_class_counts(directory):
        counts = {}
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                counts[class_name] = len(os.listdir(class_path))
        return counts

    train_counts = get_class_counts(train_dir)
    test_counts = get_class_counts(test_dir)
    
    # Create DataFrame for plotting
    df_train = pd.DataFrame(list(train_counts.items()), columns=['Class', 'Count'])
    df_train['Split'] = 'Train'
    
    df_test = pd.DataFrame(list(test_counts.items()), columns=['Class', 'Count'])
    df_test['Split'] = 'Test'
    
    df_all = pd.concat([df_train, df_test])
    
    # Print summary
    print("\n--- Training Set Class Distribution ---")
    for cls, count in sorted(train_counts.items()):
        print(f"{cls}: {count} frames")
        
    print("\n--- Test Set Class Distribution ---")
    for cls, count in sorted(test_counts.items()):
        print(f"{cls}: {count} frames")
        
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', hue='Split', data=df_all)
    plt.title('Class Distribution in UCF Crime Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(data_dir, 'class_distribution.png')
    plt.savefig(plot_path)
    print(f"\nSaved class distribution plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore dataset class distribution.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the dataset is located")
    args = parser.parse_args()
    
    explore_data(args.data_dir)
