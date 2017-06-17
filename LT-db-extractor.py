#!/usr/bin/env python

import psycopg2
import psycopg2.extras
import argparse
import json
import os
import sys
import zipfile
import paramiko
import numpy as np
import cv2

def rotateImage(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def str2bool(v):
    '''
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def options():
    parser = argparse.ArgumentParser(description='Retrieve data from a LemnaTec database.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", help="JSON config file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for results.", required=True)
    parser.add_argument("-l", "--location", help="Location of raw image, as separate file (True) or in database (False).", type = str2bool, default=False)
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        raise IOError("The directory {0} already exists!".format(args.outdir))

    return args


def main():
    # Read user options
    args = options()

    # Read the database connetion configuration file
    config = open(args.config, 'rU')
    # Load the JSON configuration data
    db = json.load(config)

    # SSH connection
    if args.location == False:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(db['hostname'], username='root', password=db['password'])
        sftp = ssh.open_sftp()

    # Make the output directory
    os.mkdir(args.outdir)

    # Create the SnapshotInfo.csv file
    csv = open(os.path.join(args.outdir, "SnapshotInfo.csv"), "w")

    # Connect to the LemnaTec database
    conn = psycopg2.connect(host=db['hostname'], user=db['username'], password=db['password'], database=db['database'])
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Get all snapshots
    snapshots = {}
    cur.execute("SELECT * FROM snapshot WHERE measurement_label = %s;", [db['experiment']])
    for row in cur:
        snapshots[row['id']] = row

    # Get all image metadata
    images = {}
    raw_images = {}
    cur.execute("SELECT * FROM snapshot INNER JOIN tiled_image ON snapshot.id = tiled_image.snapshot_id INNER JOIN tile ON tiled_image.id = tile.tiled_image_id")
    for row in cur:
        if row['snapshot_id'] in snapshots:
            image_name = row['camera_label'] + '_' + str(row['tiled_image_id']) + '_' + str(row['frame'])
            if row['snapshot_id'] in images:
                images[row['snapshot_id']].append( ( image_name, row['height'], row['width'], row['rotate_flip_type']))
            else:
                images[row['snapshot_id']] = [ ( image_name, row['height'], row['width'], row['rotate_flip_type'])]
            raw_images[image_name] = row['raw_image_oid']

    # Create SnapshotInfo.csv file
    header = ['experiment', 'id', 'plant barcode', 'car tag', 'timestamp', 'weight before', 'weight after',
              'water amount', 'completed', 'measurement label', 'tag', 'tiles']
    csv.write(','.join(map(str, header)) + '\n')

    # Stats
    total_snapshots = len(snapshots)
    total_water_jobs = 0
    total_images = 0

    for snapshot_id in snapshots.keys():
        # Reformat the completed field
        # if snapshots[snapshot_id]['completed'] == 't':
        #     snapshots[snapshot_id]['completed'] = 'true'
        # else:
        #     snapshots[snapshot_id]['completed'] = 'false'

        # Group all the output metadata
        snapshot = snapshots[snapshot_id]
        values = [db['experiment'], snapshot['id'], snapshot['id_tag'], snapshot['car_tag'],
                  snapshot['time_stamp'].strftime('%Y-%m-%d %H:%M:%S'), snapshot['weight_before'],
                  snapshot['weight_after'], snapshot['water_amount'], snapshot['completed'],
                  snapshot['measurement_label'], '']

        # If the snapshot also contains images, add them to the output
        if snapshot_id in images:
            values.append(';'.join(map(str, [iname[0] for iname in images[snapshot_id]])))
            total_images += len(images[snapshot_id])
            # Create the local directory
            snapshot_dir = os.path.join(args.outdir, "snapshot" + str(snapshot_id))
            os.mkdir(snapshot_dir)

            for image in images[snapshot_id]:
                # Copy the raw image to the local directory
                remote_dir = os.path.join("/data/pgftp", db['database'],
                                          snapshot['time_stamp'].strftime("%Y-%m-%d"), "blob" + str(raw_images[image]))
                local_file = os.path.join(snapshot_dir, "blob" + str(raw_images[image]))
                if not(args.location):
                    # if the large object/raw image is stored external to the database it can be copied by ftp
                    try:
                        sftp.get(remote_dir, local_file)
                    except IOError as e:
                        print("I/O error({0}): {1}. Offending file: {2}".format(e.errno, e.strerror, remote_dir))
                else:
                    try:
                        # if the large object/raw image is stored inside the database use the lobject function to open a connection,
                        # read the zip file then write it out to the local directory 
                        lo = psycopg2.extensions.lobject(conn, raw_images[image[0]], 'rb')
                        data = io.BytesIO( lo.read())
                        with open( os.path.join(snapshot_dir, "blob" + str(raw_images[image[0]])),'wb') as out:
                            out.write( data.read()) 
                        out.close()
                        lo.close()
                    except Exception as e:
                        # catch all exception until I understand what types of errors to expect and need to stop
                        print( e)

                if os.path.exists(local_file):
                    # Is the file a zip file?
                    if zipfile.is_zipfile(local_file):
                        zf = zipfile.ZipFile(local_file)
                        zff = zf.open("data")
                        img_str = zff.read()

                        if 'VIS' in image or 'vis' in image:
                            if len(img_str) == db['vis_height'] * db['vis_width']:
                                raw = np.fromstring(img_str, dtype=np.uint8, count=db['vis_height']*db['vis_width'])
                                raw_img = raw.reshape((db['vis_height'], db['vis_width']))
                                img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_RG2BGR)
                                cv2.imwrite(os.path.join(snapshot_dir, image + ".png"), img)
                                #os.remove(local_file)
                            else:
                                print("Warning: File {0} containing image {1} seems corrupted.".format(local_file,
                                                                                                       image))
                        elif 'NIR' in image or 'nir' in image:
                            if len(img_str) == (db['nir_height'] * db['nir_width']) * 2:
                                raw = np.fromstring(img_str, dtype=np.uint16, count=db['nir_height'] * db['nir_width'])
                                if np.max(raw) > 4096:
                                    print("Warning: max value for image {0} is greater than 4096.".format(image))
                                raw_rescale = np.multiply(raw, 16)
                                raw_img = raw_rescale.reshape((db['nir_height'], db['nir_width']))
                                cv2.imwrite(os.path.join(snapshot_dir, image + ".png"), raw_img)
                                #os.remove(local_file)
                            else:
                                print("Warning: File {0} containing image {1} seems corrupted.".format(local_file,
                                                                                                       image))
                        elif 'PSII' in image[0] or 'psII' in image[0]:
                            raw = np.fromstring(img_str, dtype=np.uint16, count=db['psII_height'] * db['psII_width'])
                            if np.max(raw) > 16384:
                                print("Warning: max value for image {0} is greater than 16384.".format(image))
                            raw_rescale = np.multiply(raw, 4)
                            raw_img = raw_rescale.reshape((db['psII_height'], db['psII_width']))
                            cv2.imwrite(os.path.join(snapshot_dir, image + ".png"), raw_img)
                            #os.remove(local_file)
                        else:
                            if len(img_str) == image[1] * image[2]:
                                raw = np.fromstring(img_str, dtype=np.uint8, count = image[1] * image[2])
                                raw_img = raw.reshape(( image[1], image[2]))
                                if 'N-TV' in image[0] or 'N0' in image[0] or 'N-' in image[0]:
                                    img = raw_img

                                else:
                                    img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_RG2BGR)
                                rotflipdict = { 0: ( 0, 0), 1: ( 270, 0), 2: ( 180, 0), 3: ( 90, 0)}
                                try:
                                    img = rotateImage( img, rotflipdict[ image[3]][0])
                                    cv2.imwrite(os.path.join(snapshot_dir, image[0] + ".png"), img)
                                except KeyError:
                                    Print( "Don't know Rotate/FlipType: {0}".format( image[3])) 
                            else:
                                print("Warning: File {0} containing image {1} seems corrupted.".format(local_file,
                                                                                                       image[0]))
                            #os.remove(local_file)
                        zff.close()
                        zf.close()
                        os.remove(local_file)
        else:
            values.append('')
            total_water_jobs += 1

        csv.write(','.join(map(str, values)) + '\n')

    cur.close()
    conn.close()
    sftp.close()
    ssh.close()

    print("Total snapshots = " + str(total_snapshots))
    print("Total water jobs = " + str(total_water_jobs))
    print("Total images = " + str(total_images))


if __name__ == '__main__':
    main()
