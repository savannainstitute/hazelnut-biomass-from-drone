import os
import argparse
from lidar_preprocessing.preprocessing import preprocess_lidar
from canopy_segmentation.segmentation import segment_canopies
from biomass_estimation.biomass import run_allometry

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: LiDAR preprocessing, canopy segmentation, biomass & carbon estimation."
    )
    parser.add_argument('--input-las', required=True, help='Input LAS file path')
    parser.add_argument('--tree-tops-shp', required=True, help='Input tree tops shapefile path')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--extent-shapefile', default=None, help='Optional extent shapefile for cropping/masking')
    parser.add_argument('--res', type=float, default=None, help='Raster resolution in meters (default: LAS point spacing)')
    args = parser.parse_args()

    # Step 1: Preprocess LiDAR
    print("Preprocessing LiDAR...")
    pre = preprocess_lidar(args.input_las, args.output_dir, res=args.res)
    chm_path = pre["chm"]
    print(f"CHM created at: {chm_path}")

    # Step 2: Canopy Segmentation
    print("Segmenting canopies...")
    seg = segment_canopies(
        chm_path=chm_path,
        tree_tops_shp=args.tree_tops_shp,
        output_dir=args.output_dir,
        extent_shapefile=args.extent_shapefile
    )
    if seg is None:
        print("Segmentation failed.")
        return
    print(f"Canopy segments saved to: {seg}")

    # Step 3: Biomass & Carbon Estimation
    print("Estimating biomass and carbon...")
    prefix = os.path.splitext(os.path.basename(chm_path))[0].replace('_chm', '')
    polygons_path = os.path.join(args.output_dir, f"{prefix}_segments.shp")
    output_shp = os.path.join(args.output_dir, f"{prefix}_biomass_results.shp")
    output_csv = os.path.join(args.output_dir, f"{prefix}_biomass_results.csv")
    run_allometry(polygons_path, chm_path, output_shp, output_csv)
    print("Processing complete.")
    print(f"Results saved to: {output_shp} and {output_csv}")

if __name__ == "__main__":
    main()