import json
from tuples import TopicLocation


def topTermsAsList(terms):
    '''
        Return some terms from a list of terms and weights

                Parameters:
                        terms (List(List(String))): terms and weights list

                Returns:
                        List(String): terms as a list
    '''
    l = []
    for i in range(len(terms)):
        l.append(terms[i][0])
    return l


def topTermsAsString(terms):
    '''
        Convert some terms to a string separated by commas

                Parameters:
                        terms (List(String)): terms to join

                Returns:
                        String: terms separated by a comma
    '''
    return ','.join(topTermsAsList(terms))


def exportSurfaceToGrid(surf, out_file_name):
    '''
        Export surface as ASCII Grid data

                Parameters:
                        surf (Tuple): Tuple as returned by computeSurface
                        out_file (String): ESRI Grid ASCII export file
    '''

    with open(out_file_name, 'wb') as f:
        f.write(bytes(
            f'NROWS {surf[0].shape[0]}\nNCOLS {surf[0].shape[1]}\nXLLCENTER 0\nYLLCENTER 0\nDX {surf[1]}\nDY {surf[1]}\nNODATA_VALUE -1\n',
            'UTF-8'))

        for r in range(surf[0].shape[0]):
            for c in range(surf[0].shape[1]):
                f.write(bytes(f' {(surf[0][r, c]):9.3f}', 'UTF-8'))
            f.write(bytes('\n', 'UTF-8'))


def exportTopicLocationToGeoJSON(tl, out_file, x_scale, y_scale):
    '''
        Write topic locationa as a GeoJSON file to out_file

                Parameters:
                        tl (List(TopicLocation)): locations to export
                        out_file (String): GeoJSON output file
                        x_scale (float): scale to apply to X coordinates
                        y_scale (float): scale to apply to y coordinates

                Returns:
                        List(TopicLocation): features read from the inpout file
    '''
    geo = {'type': 'FeatureCollection', 'features': []}
    for i in range(0, len(tl)):
        geo['features'].append({'type': 'Feature',
                                'geometry': {'type': 'Point',
                                             'coordinates': [tl[i].x * x_scale, tl[i].y * y_scale]
                                             },
                                'properties': {'t': tl[i].t,
                                               'label': tl[i].label,
                                               'top_terms': topTermsAsString(tl[i].top_terms),
                                               'n': tl[i].n,
                                               'topic_id': tl[i].id,
                                               'x': round(tl[i].x, 6),
                                               'y': round(tl[i].y, 6)
                                               }
                                })
    with open(out_file, 'w') as f:
        f.write(json.dumps(geo))


def importTopicLocationFromGeoJSON(in_file_name):
    '''
        Read topic location data from a GeoJSON file

                Parameters:
                        in_file (String): GeoJSON input file

                Returns:
                        List(TopicLocation): features read from the inpout file
    '''

    def extractTLfromFeature(feat):
        # feat['properties']['topic_id']
        return TopicLocation(
            x=feat['geometry']['coordinates'][0],
            y=feat['geometry']['coordinates'][1],
            t=feat['properties']['t'],
            id=None,
            n=feat['properties']['n'],
            label=feat['properties']['label'],
            top_terms=feat['properties']['top_terms']
        )

    with open(in_file_name, 'r') as f:
        features = json.load(f)['features']

        return [extractTLfromFeature(feat) for feat in features]
