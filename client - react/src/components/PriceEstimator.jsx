import React, { useState, useEffect } from 'react';
import { getLocationNames, predictHomePrice } from '../services/api';
import './PriceEstimator.css';

const PriceEstimator = () => {
  const [sqft, setSqft] = useState(1000);
  const [bhk, setBhk] = useState(2);
  const [bathrooms, setBathrooms] = useState(2);
  const [location, setLocation] = useState('');
  const [locations, setLocations] = useState([]);
  const [estimatedPrice, setEstimatedPrice] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchLocations();
  }, []);

  const fetchLocations = async () => {
    try {
      const data = await getLocationNames();
      if (data && data.locations) {
        setLocations(data.locations);
      }
    } catch (err) {
      setError('Failed to load locations');
      console.error(err);
    }
  };

  const handleEstimatePrice = async () => {
    if (!location) {
      setError('Please select a location');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await predictHomePrice({
        total_sqft: parseFloat(sqft),
        bhk: bhk,
        bath: bathrooms,
        location: location
      });

      if (data && data.estimated_price) {
        setEstimatedPrice(`${data.estimated_price} Lakh`);
      }
    } catch (err) {
      setError('Failed to estimate price. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="background-img"></div>
      <div className="form-container">
        <div className="form">
          <h2>Area (Square Feet)</h2>
          <input
            className="area"
            type="number"
            value={sqft}
            onChange={(e) => setSqft(e.target.value)}
          />

          <h2>BHK</h2>
          <div className="switch-field">
            {[1, 2, 3, 4, 5].map((value) => (
              <React.Fragment key={value}>
                <input
                  type="radio"
                  id={`radio-bhk-${value}`}
                  name="bhk"
                  value={value}
                  checked={bhk === value}
                  onChange={() => setBhk(value)}
                />
                <label htmlFor={`radio-bhk-${value}`}>{value}</label>
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="form">
          <h2>Bath</h2>
          <div className="switch-field">
            {[1, 2, 3, 4, 5].map((value) => (
              <React.Fragment key={value}>
                <input
                  type="radio"
                  id={`radio-bath-${value}`}
                  name="bathrooms"
                  value={value}
                  checked={bathrooms === value}
                  onChange={() => setBathrooms(value)}
                />
                <label htmlFor={`radio-bath-${value}`}>{value}</label>
              </React.Fragment>
            ))}
          </div>

          <h2>Location</h2>
          <div>
            <select
              className="location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            >
              <option value="">Choose a Location</option>
              {locations.map((loc, index) => (
                <option key={index} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
          </div>

          <button
            className="submit"
            onClick={handleEstimatePrice}
            disabled={loading}
          >
            {loading ? 'Estimating...' : 'Estimate Price'}
          </button>

          {error && (
            <div className="result" style={{ background: '#ff6b6b' }}>
              <h2>{error}</h2>
            </div>
          )}

          {estimatedPrice && !error && (
            <div className="result">
              <h2>{estimatedPrice}</h2>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default PriceEstimator;