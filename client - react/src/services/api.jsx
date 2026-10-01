import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5000';
// Use this for nginx: const API_BASE_URL = '/api';

export const getLocationNames = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/get_location_names`);
    return response.data;
  } catch (error) {
    console.error('Error fetching locations:', error);
    throw error;
  }
};

export const predictHomePrice = async (data) => {
  try {
    // Convert data to FormData to match Flask's request.form expectation
    const formData = new FormData();
    formData.append('total_sqft', data.total_sqft);
    formData.append('bhk', data.bhk);
    formData.append('bath', data.bath);
    formData.append('location', data.location);

    const response = await axios.post(`${API_BASE_URL}/predict_home_price`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error predicting price:', error);
    throw error;
  }
};