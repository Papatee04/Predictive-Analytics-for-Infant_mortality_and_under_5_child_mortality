document.addEventListener('DOMContentLoaded', function() {
    const updateAssessmentBtn = document.getElementById('updateAssessmentBtn');
    const assessmentId = updateAssessmentBtn.dataset.assessmentId;
    const deleteBtn = document.querySelector('[data-delete-assessment]');

    // Update Assessment Modal
    function createUpdateModal() {
        const modalHtml = `
        <div id="updateModal" class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div class="bg-white rounded-lg shadow-xl w-96 p-6 relative">
                <button id="closeUpdateModal" class="absolute top-3 right-3 text-gray-500 hover:text-gray-800">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
                <h2 class="text-2xl font-bold mb-4 text-teal-600">Update Assessment</h2>
                <form id="updateAssessmentForm" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Status</label>
                        <select name="status" class="w-full p-2 border rounded-md focus:ring-2 focus:ring-teal-500">
                            <option value="active">Active</option>
                            <option value="resolved">Resolved</option>
                            <option value="inactive">Inactive</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Intervention Outcome</label>
                        <select name="intervention_outcome" class="w-full p-2 border rounded-md focus:ring-2 focus:ring-teal-500">
                            <option value="">Select Outcome</option>
                            <option value="successful">Successful</option>
                            <option value="unsuccessful">Unsuccessful</option>
                        </select>
                    </div>
                    <div class="flex justify-end space-x-3">
                        <button type="button" id="cancelUpdateBtn" class="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300">
                            Cancel
                        </button>
                        <button type="submit" class="px-4 py-2 bg-teal-600 text-white rounded hover:bg-teal-700">
                            Update
                        </button>
                    </div>
                </form>
            </div>
        </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        setupUpdateModalEvents();
    }

    function setupUpdateModalEvents() {
        const modal = document.getElementById('updateModal');
        const closeBtn = document.getElementById('closeUpdateModal');
        const cancelBtn = document.getElementById('cancelUpdateBtn');
        const form = document.getElementById('updateAssessmentForm');

        // Close modal functions
        const closeModal = () => modal.remove();
        [closeBtn, cancelBtn].forEach(el => el.addEventListener('click', closeModal));

        // Form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(form);

            try {
                const response = await fetch(`/assessments/${assessmentId}/update/`, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                    }
                });

                if (response.ok) {
                    // Refresh the page or update UI
                    window.location.reload();
                } else {
                    throw new Error('Update failed');
                }
            } catch (error) {
                alert('Failed to update assessment. Please try again.');
            }
        });
    }

    // Delete Confirmation Modal
    function createDeleteModal() {
        const modalHtml = `
        <div id="deleteModal" class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div class="bg-white rounded-lg shadow-xl w-96 p-6 text-center">
                <div class="mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 text-red-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <h2 class="text-2xl font-bold text-gray-800 mb-2">Confirm Deletion</h2>
                    <p class="text-gray-600 mb-4">Are you sure you want to delete this assessment? This action cannot be undone.</p>
                </div>
                <div class="flex justify-center space-x-4">
                    <button id="cancelDeleteBtn" class="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300">
                        Cancel
                    </button>
                    <button id="confirmDeleteBtn" class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                        Delete Assessment
                    </button>
                </div>
            </div>
        </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        setupDeleteModalEvents();
    }

    function setupDeleteModalEvents() {
        const modal = document.getElementById('deleteModal');
        const cancelBtn = document.getElementById('cancelDeleteBtn');
        const confirmBtn = document.getElementById('confirmDeleteBtn');

        // Close modal function
        const closeModal = () => modal.remove();
        cancelBtn.addEventListener('click', closeModal);

        // Confirm delete
        confirmBtn.addEventListener('click', async function() {
            try {
                const response = await fetch(`/assessments/${assessmentId}/delete/`, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                    }
                });

                if (response.ok) {
                    // Redirect to assessments management page
                    window.location.href = '/assessments/';
                } else {
                    throw new Error('Deletion failed');
                }
            } catch (error) {
                alert('Failed to delete assessment. Please try again.');
            }
        });
    }

    // Event Listeners
    updateAssessmentBtn.addEventListener('click', createUpdateModal);
    deleteBtn.addEventListener('click', createDeleteModal);
});