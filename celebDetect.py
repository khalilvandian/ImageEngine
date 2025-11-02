import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import shutil
    return os, pd, plt, shutil, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ### Loading the CelebA Identity Dataset

    The `identity_CelebA.txt` file contains the mapping between image filenames and the celebrity identity they belong to. We'll load this file into a pandas DataFrame for further analysis. The file is space-separated and does not contain a header.
    """)
    return


@app.cell
def _():
    images_path = "D:/Projects/ImageEngine/Images/New folder/Img-20251102T111410Z-1-001/Img/img_celeba.7z/img_celeba.7z/img_celeba"
    identity_file_path = 'images/Anno/identity_CelebA.txt'
    return identity_file_path, images_path


@app.cell
def _(identity_file_path, pd):
    # Read the space-separated file into a pandas DataFrame
    # We assign column names 'image_id' and 'celebrity_id'
    identity_df = pd.read_csv(
        identity_file_path,
        sep=' ',
        header=None,
        names=['image_id', 'celebrity_id']
    )

    identity_df.head()
    return (identity_df,)


@app.cell
def _(identity_df, plt, sns):
    # Calculate the number of images for each celebrity
    celebrity_counts = identity_df['celebrity_id'].value_counts()

    # Print the total number of unique celebrities (celebrity id count)
    print(f"Total number of unique celebrities: {len(celebrity_counts)}")

    # Display the counts for the top 15 celebrities (count of each id)
    print("\nImage counts for the top 15 celebrities:")
    print(celebrity_counts.head(15))

    # Visualize the distribution of the counts
    plt.figure(figsize=(12, 6))
    sns.histplot(celebrity_counts, bins=50, kde=True, log_scale=(False, True))
    plt.title('Distribution of Image Counts per Celebrity')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of Celebrities (Log Scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca()
    return


@app.cell
def _(identity_df):
    image_id_to_find = "001467.jpg"
    celebrity_id_for_image = identity_df[identity_df['image_id'] == image_id_to_find]

    celebrity_id_for_image
    return


@app.cell
def _(identity_df):
    images_for_celebrity_3899 = identity_df[identity_df['celebrity_id'] == 3899]
    images_for_celebrity_3899
    return


@app.cell
def _(identity_df, images_path, os, shutil):
    # Filter the DataFrame to exclude celebrity 3899
    not_3899_df = identity_df[identity_df['celebrity_id'] != 3899]

    # Randomly sample 50 images from the filtered DataFrame
    # Using random_state for reproducibility
    random_sample_not_3899 = not_3899_df.sample(n=50, random_state=42)

    # Define the destination directory for the "not 3899" images
    destination_dir_not_3899 = 'images/not_3899'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir_not_3899, exist_ok=True)

    # Get the list of image filenames from the random sample
    image_list_not_3899 = random_sample_not_3899['image_id'].tolist()

    # Loop through the image list and copy each file
    copied_count_not_3899 = 0
    for image_file in image_list_not_3899:
        source_file_path = os.path.join(images_path, image_file)
        destination_file_path = os.path.join(destination_dir_not_3899, image_file)
    
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, destination_file_path)
            copied_count_not_3899 += 1
        else:
            print(f"File not found, skipping: {source_file_path}")

    print(f"Successfully copied {copied_count_not_3899} of {len(image_list_not_3899)} random images to '{destination_dir_not_3899}'")
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
