import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from .generator import Nonogram
from .solver import NonogramSolver

app = Flask(__name__)
CORS(app)

app.json_encoder = NumpyEncoder

# Use environment variable for upload folder to support different deployment environments
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

active_nonograms = {}

# Add a route to help identify the API and provide basic information
@app.route('/')
def index():
    return jsonify({
        'name': 'NonogramSolver API',
        'status': 'online',
        'endpoints': {
            '/upload': 'POST - Upload an image and convert to nonogram',
            '/solve/<nonogram_id>': 'POST - Start solving a nonogram',
            '/next-iteration/<nonogram_id>': 'GET - Get the next iteration in solving process',
            '/completed-solution/<nonogram_id>': 'GET - Get the complete solution',
            '/clean/<nonogram_id>': 'DELETE - Clean up resources'
        }
    })

def array_to_base64_img(array):
    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap='binary', interpolation='nearest')
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    size = request.form.get('size', '15,15')
    try:
        width, height = map(int, size.split(','))
    except:
        return jsonify({'error': 'Invalid size format. Use "width,height"'}), 400
    
    threshold = int(request.form.get('threshold', '128'))
    
    unique_id = str(uuid.uuid4())
    filename = unique_id + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    nonogram = Nonogram((width, height))
    nonogram.image_to_nonogram(filepath, threshold=threshold)
    
    row_clues, col_clues = nonogram.generate_clues()
    
    active_nonograms[unique_id] = {
        'nonogram': nonogram,
        'filepath': filepath,
        'solver': None,
        'iterations': [],
    }
    
    original_img = array_to_base64_img(nonogram.nonogram)
    
    return jsonify({
        'id': unique_id,
        'size': f"{width}x{height}",
        'row_clues': row_clues,
        'col_clues': col_clues,
        'grid_data': nonogram.nonogram.tolist(),
        'image': original_img,
        'cell_states': {
            '1': 'filled',
            '0': 'unknown',
            '-1': 'empty'
        }
    }), 200

@app.route('/solve/<nonogram_id>', methods=['POST'])
def solve_nonogram(nonogram_id):
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    nonogram = nonogram_data['nonogram']
    
    row_clues, col_clues = nonogram.generate_clues()
    solver = NonogramSolver(row_clues, col_clues)
    
    nonogram_data['solver'] = solver
    nonogram_data['iterations'] = []
    
    original_update_grid = solver.update_grid
    
    def update_grid_with_capture():
        original_update_grid()
        nonogram_data['iterations'].append(np.copy(solver.nonogram))
    
    solver.update_grid = update_grid_with_capture
    
    solver.get_all_solutions()
    solver.update_grid()
    
    if nonogram_data['iterations']:
        first_iteration = array_to_base64_img(nonogram_data['iterations'][0])
        return jsonify({
            'iteration': 1,
            'total_iterations': len(nonogram_data['iterations']),
            'grid_data': nonogram_data['iterations'][0].tolist(),
            'image': first_iteration,
            'completed': False
        }), 200
    else:
        return jsonify({'error': 'Failed to start solving'}), 500

@app.route('/next-iteration/<nonogram_id>', methods=['GET'])
def next_iteration(nonogram_id):
    if nonogram_id not in active_nonograms:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], nonogram_id)
        if os.path.exists(file_path + '.jpeg') or os.path.exists(file_path + '.jpg') or os.path.exists(file_path + '.png'):
            return jsonify({
                'error': 'Nonogram exists as a file but not in active memory. Try uploading again or initializing solver.',
                'status': 'file_exists_but_not_active'
            }), 404
        return jsonify({'error': 'Nonogram not found', 'id': nonogram_id}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    solver = nonogram_data['solver']
    
    if not solver:
        return jsonify({'error': 'Solver not initialized. Call /solve endpoint first.'}), 400
    
    try:
        current_iteration = int(request.args.get('iteration', 0))
    except ValueError:
        return jsonify({'error': 'Invalid iteration parameter. Must be a number.'}), 400
    
    print(f"Requested iteration {current_iteration} for nonogram {nonogram_id}")
    print(f"Available iterations: {len(nonogram_data['iterations'])}")
    
    if current_iteration >= len(nonogram_data['iterations']):
        if not np.all(solver.nonogram != 0):
            if hasattr(solver, 'last_nonogram') and np.array_equal(solver.last_nonogram, solver.nonogram):
                solver.last_nonogram = np.copy(solver.nonogram)
                guess_made = False
                solver.save_before_guess()
                
                for i in range(solver.m):
                    for j in range(solver.n):
                        if solver.nonogram[i, j] == 0:
                            solver.nonogram[i, j] = 1
                            solver.update_grid()
                            if not np.array_equal(solver.last_nonogram, solver.nonogram):
                                guess_made = True
                                break
                            else:
                                solver.reset_before_guess()
                                solver.nonogram[i, j] = -1
                                solver.update_grid()
                                if not np.array_equal(solver.last_nonogram, solver.nonogram):
                                    guess_made = True
                                    break
                            solver.reset_before_guess()
                    if guess_made:
                        break
            else:
                solver.last_nonogram = np.copy(solver.nonogram)
                solver.update_grid()
            
            print(f"Generated iteration {len(nonogram_data['iterations'])-1}")
    
    if current_iteration < len(nonogram_data['iterations']):
        iteration_img = array_to_base64_img(nonogram_data['iterations'][current_iteration])
        completed = bool(np.all(nonogram_data['iterations'][current_iteration] != 0))
        
        return jsonify({
            'iteration': current_iteration + 1,
            'total_iterations': len(nonogram_data['iterations']),
            'grid_data': nonogram_data['iterations'][current_iteration].tolist(),
            'image': iteration_img,
            'completed': completed
        }), 200
    else:
        latest_iteration = len(nonogram_data['iterations']) - 1
        if latest_iteration >= 0:
            iteration_img = array_to_base64_img(nonogram_data['iterations'][latest_iteration])
            completed = bool(np.all(nonogram_data['iterations'][latest_iteration] != 0))
            
            return jsonify({
                'iteration': latest_iteration + 1,
                'total_iterations': len(nonogram_data['iterations']),
                'grid_data': nonogram_data['iterations'][latest_iteration].tolist(),
                'image': iteration_img,
                'completed': completed,
                'note': f"Requested iteration {current_iteration} not available, returning latest (iteration {latest_iteration})"
            }), 200
        else:
            return jsonify({
                'error': 'No iterations available',
                'requested': current_iteration,
                'available': len(nonogram_data['iterations']),
                'status': 'no_iterations'
            }), 404

@app.route('/completed-solution/<nonogram_id>', methods=['GET'])
def completed_solution(nonogram_id):
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    solver = nonogram_data['solver']
    
    if not solver:
        return jsonify({'error': 'Solver not initialized'}), 400
    
    if not np.all(solver.nonogram != 0):
        iterations_before = len(nonogram_data['iterations'])
        
        while not np.all(solver.nonogram != 0):
            solver.update_grid()
            if hasattr(solver, 'last_nonogram') and np.array_equal(solver.last_nonogram, solver.nonogram):
                guess_made = False
                solver.save_before_guess()
                
                for i in range(solver.m):
                    for j in range(solver.n):
                        if solver.nonogram[i, j] == 0:
                            solver.nonogram[i, j] = 1
                            solver.update_grid()
                            if not np.array_equal(solver.last_nonogram, solver.nonogram):
                                guess_made = True
                                break
                            else:
                                solver.reset_before_guess()
                                solver.nonogram[i, j] = -1
                                solver.update_grid()
                                if not np.array_equal(solver.last_nonogram, solver.nonogram):
                                    guess_made = True
                                    break
                            solver.reset_before_guess()
                    if guess_made:
                        break
                if not guess_made:
                    break
            
            solver.last_nonogram = np.copy(solver.nonogram)
    
    solution_img = array_to_base64_img(solver.nonogram)
    
    original = nonogram_data['nonogram'].original
    original_img = array_to_base64_img(original)
    
    difference_img = None
    difference_grid = None
    if original is not None and original.shape == solver.nonogram.shape:
        difference = solver.nonogram - original
        difference_img = array_to_base64_img(difference)
        difference_grid = difference.tolist()
    
    return jsonify({
        'solution_grid': solver.nonogram.tolist(),
        'solution': solution_img,
        'original_grid': original.tolist() if original is not None else None,
        'original': original_img,
        'difference_grid': difference_grid,
        'difference': difference_img,
        'total_iterations': len(nonogram_data['iterations'])
    }), 200

@app.route('/clean/<nonogram_id>', methods=['DELETE'])
def clean_nonogram(nonogram_id):
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    filepath = active_nonograms[nonogram_id]['filepath']
    if os.path.exists(filepath):
        os.remove(filepath)
    
    del active_nonograms[nonogram_id]
    
    return jsonify({'success': True}), 200

if __name__ == '__main__':
    app.run(debug=True)