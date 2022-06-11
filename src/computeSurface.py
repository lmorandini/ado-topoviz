import sys, os, json, numpy as np, configargparse, math, glob

sys.path.append('../src')

from logger import logger
from export_lib import exportSurfaceToGrid, importTopicLocationFromGeoJSON
from tqdm import tqdm


def computeSurface(locs, n_rows, max_dist, z_scale):
    '''
        Compute surface from a set of features
        :param locs (List(TopicLocation)): locations to interpolate from
        :param n_rows (int): n. of rows of the surface
        :param max_dist (int): maximum distance for KDE expressed in distance between intervals
        :param z_scale (float): rescaling factor of Z values
        :return: (numpy.array. size): Tuple(2D array of grid values, grid cell size)
    '''

    def kde_quartic(d, h, z):
        '''
        Compute the Quartic kernel Density Estimator of value z
        :param d (float): distance of z from a give point
        :param h (float): max distance that affects the KDE computation
        :param z (float): value to calculate the KDE of
        :return: (float) Warped Euclidean distance
        '''
        if d <= h:
            return ((15 / 16) * (1 - (d / h) ** 2) ** 2) * z
        else:
            return 0

    def warped_euclidian_dist(p1, p2, yx):
        '''
        Compute warped Euclidiand distance between points p1 and p2
        :param p1 (float, float): first point
        :param p2 (float, float): second point
        :param yx (float): factor to "stretch" the X distance of
        :return: (float) KDE value
        '''
        return math.sqrt(((p1[0] - p2[0]) ** 2) / yx + (p1[1] - p2[1]) ** 2)

    # Extracts coordinates and computer boundaries
    x = np.array([loc.x for loc in locs])
    y = np.array([loc.y for loc in locs])
    z = np.array([loc.n for loc in locs])
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # Computes the cell size based on the number of unique timestamps
    g = (ymax - ymin) / n_rows

    n_cols = int((xmax - xmin) / g)
    gz = np.ndarray([n_rows, n_cols])
    n = len(locs)
    t = len(set([loc.t for loc in locs]))

    # Ratio of "stratching" of x distance
    x_warp = n / (t ** 2)

    # Max distance for KDE expressed in surface units
    max_dist_u = max_dist * g * (n_cols / t)

    logger.warning(
        f'Computing surface ({n_rows},{n_cols}), cell size:{g:.2f} n:{n} t:{t} n per t:{(n / t):.2f} x:{xmax}-{xmin} y:{ymax}-{ymin} max dist:{max_dist_u:.2f} X warp:{x_warp:.2f}')

    for r in tqdm(range(n_rows)):
        for c in range(n_cols):
            gz[r, c] = 0
            for i in range(len(x)):
                gz[r, c] += kde_quartic(
                    warped_euclidian_dist((c * g, r * g), (x[i], y[i]), x_warp),
                    max_dist_u,
                    z[i]
                )
            gz[r, c] = gz[r, c] * z_scale
    return (gz, g)


def main():
    argp = configargparse.ArgParser()
    argp.add('--input_dir', required=False, type=str, env_var='INPUT_DIR',
             help='directory holding input data')
    argp.add('--output_dir', required=True, type=str, env_var='OUTPUT_DIR',
             help='directory holding output data')
    argp.add('--model_name', required=True, type=str, env_var='MODEL_NAME',
             help='output model name')
    argp.add('--n_rows', required=False, nargs='?', const=1, default=100, type=int, env_var='N_ROWS',
             help='N. of rows of the surface grid')
    argp.add('--max_dist', required=False, nargs='?', const=1, default=100, type=float, env_var='MAX_CELLS_DIST',
             help='Max distance (expressed in distance between consecutive timee intervals) the KDE uses')
    argp.add('--z_scale', required=False, nargs='?', const=1, default=1, type=float, env_var='Z_SCALE',
             help='Z rescaling')

    settings = argp.parse_known_args()[0]
    geojson_files = glob.glob(f'{settings.input_dir}/{settings.model_name}.topiclocation.geojson')
    geojson_files.sort()

    for file_name in geojson_files:
        logger.warning(f'Read file {file_name}')

        # Export surface as ASCII Grid file
        grid_file_name = f'{settings.output_dir}/{settings.model_name}.surface.asc'
        exportSurfaceToGrid(
            computeSurface(
                importTopicLocationFromGeoJSON(file_name),
                settings.n_rows, settings.max_dist, settings.z_scale
            ),
            grid_file_name)
        logger.warning(f'Written {grid_file_name}')


if __name__ == '__main__':
    # WARNING log level is used to avoid gensim printing out very verbose logs at INFO level
    logger.warning(f'Started {__file__}')
    main()
    logger.warning(f'ended')
