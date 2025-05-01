import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Custom JSON encoder to handle NumPy types
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
CORS(app)  # Enable CORS for all routes

# Configure Flask to use our custom JSON encoder
app.json_encoder = NumpyEncoder

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Store active nonograms in memory
active_nonograms = {}

def array_to_base64_img(array):
    """Convert numpy array to base64 encoded PNG image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap='binary', interpolation='nearest')
    plt.axis('off')
    
    # Save image to BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Get the image as base64 string
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and generate nonogram"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get size from request or use default
    size = request.form.get('size', '15,15')
    try:
        width, height = map(int, size.split(','))
    except:
        return jsonify({'error': 'Invalid size format. Use "width,height"'}), 400
    
    threshold = int(request.form.get('threshold', '128'))
    
    # Save file with unique name
    unique_id = str(uuid.uuid4())
    filename = unique_id + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Create nonogram from image
    nonogram = Nonogram((width, height))
    nonogram.image_to_nonogram(filepath, threshold=threshold)
    
    # Generate clues
    row_clues, col_clues = nonogram.generate_clues()
    
    # Store the nonogram in memory
    active_nonograms[unique_id] = {
        'nonogram': nonogram,
        'filepath': filepath,
        'solver': None,
        'iterations': [],
    }
    
    # Convert nonogram grid to base64 image
    original_img = array_to_base64_img(nonogram.nonogram)
    
    return jsonify({
        'id': unique_id,
        'size': f"{width}x{height}",
        'row_clues': row_clues,
        'col_clues': col_clues,
        'image': original_img
    }), 200

@app.route('/solve/<nonogram_id>', methods=['POST'])
def solve_nonogram(nonogram_id):
    """Start solving the nonogram and return the first iteration"""
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    nonogram = nonogram_data['nonogram']
    
    # Setup solver
    row_clues, col_clues = nonogram.generate_clues()
    solver = NonogramSolver(row_clues, col_clues)
    
    # Store solver for future iterations
    nonogram_data['solver'] = solver
    nonogram_data['iterations'] = []
    
    # Modify solver to capture iterations
    original_update_grid = solver.update_grid
    
    def update_grid_with_capture():
        original_update_grid()
        # Capture the current state
        nonogram_data['iterations'].append(np.copy(solver.nonogram))
    
    # Replace the method with our capturing version
    solver.update_grid = update_grid_with_capture
    
    # Start solving
    solver.get_all_solutions()
    # Run first iteration
    solver.update_grid()
    
    # Get the first iteration result
    if nonogram_data['iterations']:
        first_iteration = array_to_base64_img(nonogram_data['iterations'][0])
        return jsonify({
            'iteration': 1,
            'total_iterations': len(nonogram_data['iterations']),
            'image': first_iteration,
            'completed': False
        }), 200
    else:
        return jsonify({'error': 'Failed to start solving'}), 500

@app.route('/next-iteration/<nonogram_id>', methods=['GET'])
def next_iteration(nonogram_id):
    """Get the next iteration of the solving process"""
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    solver = nonogram_data['solver']
    
    if not solver:
        return jsonify({'error': 'Solver not initialized'}), 400
    
    # Get current iteration number from query param
    current_iteration = int(request.args.get('iteration', 0))
    
    if current_iteration >= len(nonogram_data['iterations']):
        # Run next iteration if needed
        if not np.all(solver.nonogram != 0):
            # Check if there's no progress (needs guessing)
            if hasattr(solver, 'last_nonogram') and np.array_equal(solver.last_nonogram, solver.nonogram):
                # The solver would now make guesses, which requires special handling
                solver.last_nonogram = np.copy(solver.nonogram)
                # Do the guessing logic
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
                # Normal iteration
                solver.last_nonogram = np.copy(solver.nonogram)
                solver.update_grid()
    
    # If we have the requested iteration
    if current_iteration < len(nonogram_data['iterations']):
        iteration_img = array_to_base64_img(nonogram_data['iterations'][current_iteration])
        # Convert np.bool_ to Python bool to ensure JSON serialization works
        completed = bool(np.all(nonogram_data['iterations'][current_iteration] != 0))
        
        return jsonify({
            'iteration': current_iteration + 1,
            'total_iterations': len(nonogram_data['iterations']),
            'image': iteration_img,
            'completed': completed
        }), 200
    else:
        return jsonify({'error': 'No more iterations available'}), 404

@app.route('/completed-solution/<nonogram_id>', methods=['GET'])
def completed_solution(nonogram_id):
    """Get the final solution of the nonogram"""
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    nonogram_data = active_nonograms[nonogram_id]
    solver = nonogram_data['solver']
    
    if not solver:
        return jsonify({'error': 'Solver not initialized'}), 400
    
    # Complete the solution if not already completed
    if not np.all(solver.nonogram != 0):
        # Save the current state for progress reporting
        iterations_before = len(nonogram_data['iterations'])
        
        # Run the solver to completion
        while not np.all(solver.nonogram != 0):
            solver.update_grid()
            if hasattr(solver, 'last_nonogram') and np.array_equal(solver.last_nonogram, solver.nonogram):
                # Do guessing
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
                    break  # Cannot make progress, exit
            
            solver.last_nonogram = np.copy(solver.nonogram)
    
    # Get the final solution
    solution_img = array_to_base64_img(solver.nonogram)
    
    # Get the original image for comparison
    original = nonogram_data['nonogram'].original
    original_img = array_to_base64_img(original)
    
    # Calculate difference if original is available
    difference_img = None
    if original is not None and original.shape == solver.nonogram.shape:
        difference = solver.nonogram - original
        difference_img = array_to_base64_img(difference)
    
    return jsonify({
        'solution': solution_img,
        'original': original_img,
        'difference': difference_img,
        'total_iterations': len(nonogram_data['iterations'])
    }), 200

@app.route('/clean/<nonogram_id>', methods=['DELETE'])
def clean_nonogram(nonogram_id):
    """Remove nonogram data and uploaded file"""
    if nonogram_id not in active_nonograms:
        return jsonify({'error': 'Nonogram not found'}), 404
    
    # Delete the uploaded file
    filepath = active_nonograms[nonogram_id]['filepath']
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Remove from memory
    del active_nonograms[nonogram_id]
    
    return jsonify({'success': True}), 200

if __name__ == '__main__':
    app.run(debug=True)