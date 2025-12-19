import React, { useState } from 'react';

interface ProteinProperties {
  f1_total_surface_area: number | null;
  f2_non_polar_exposed_area: number | null;
  f3_fractional_area_exposed_non_polar_residue: number | null;
  f4_fractional_area_exposed_non_polar_part_residue: number | null;
  f5_molecular_mass_weighted_exposed_area: number | null;
  f6_average_deviation_from_standard_exposed_area: number | null;
  f7_euclidian_distance: number | null;
  f8_secondary_structure_penalty: number | null;
  f9_spatial_distribution_constraints: number | null;
}

const initialProperties: ProteinProperties = {
  f1_total_surface_area: null,
  f2_non_polar_exposed_area: null,
  f3_fractional_area_exposed_non_polar_residue: null,
  f4_fractional_area_exposed_non_polar_part_residue: null,
  f5_molecular_mass_weighted_exposed_area: null,
  f6_average_deviation_from_standard_exposed_area: null,
  f7_euclidian_distance: null,
  f8_secondary_structure_penalty: null,
  f9_spatial_distribution_constraints: null,
};

const propertyLabels: Record<keyof ProteinProperties, string> = {
  f1_total_surface_area: 'F1: Total Surface Area',
  f2_non_polar_exposed_area: 'F2: Non Polar Exposed Area',
  f3_fractional_area_exposed_non_polar_residue: 'F3: Fractional Area of Exposed Non Polar Residue',
  f4_fractional_area_exposed_non_polar_part_residue: 'F4: Fractional Area of Exposed Non Polar Part of Residue',
  f5_molecular_mass_weighted_exposed_area: 'F5: Molecular Mass Weighted Exposed Area',
  f6_average_deviation_from_standard_exposed_area: 'F6: Average Deviation from Standard Exposed Area of Residue',
  f7_euclidian_distance: 'F7: Euclidian Distance',
  f8_secondary_structure_penalty: 'F8: Secondary Structure Penalty',
  f9_spatial_distribution_constraints: 'F9: Spatial Distribution Constraints (N,K Value)',
};

interface InputFormProps {
  onPredict?: (gar: number, garExp: number, mae: number) => void;
}

export const InputForm: React.FC<InputFormProps> = ({ onPredict } ) => {
  const [properties, setProperties] = useState<ProteinProperties>(initialProperties);
  const [errors, setErrors] = useState<Record<keyof ProteinProperties, boolean>>({
    f1_total_surface_area: false,
    f2_non_polar_exposed_area: false,
    f3_fractional_area_exposed_non_polar_residue: false,
    f4_fractional_area_exposed_non_polar_part_residue: false,
    f5_molecular_mass_weighted_exposed_area: false,
    f6_average_deviation_from_standard_exposed_area: false,
    f7_euclidian_distance: false,
    f8_secondary_structure_penalty: false,
    f9_spatial_distribution_constraints: false,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const parsedValue = parseFloat(value);
    const finalValue = isNaN(parsedValue) ? 0 : Math.max(0, parsedValue);
    
    setProperties((prev) => ({
      ...prev,
      [name]: finalValue,
    }));

    // Validate: value must be greater than 0
    setErrors((prev) => ({
      ...prev,
      [name]: finalValue <= 0,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate all fields
    const newErrors: Record<keyof ProteinProperties, boolean> = {} as any;
    let hasError = false;

    (Object.keys(properties) as Array<keyof ProteinProperties>).forEach((key) => {
      const isInvalid = properties[key]! <= 0;
      newErrors[key] = isInvalid;
      if (isInvalid) hasError = true;
    });

    setErrors(newErrors);

    if (hasError) {
      alert('All protein properties must be greater than 0');
      return;
    }

    // If validation passes, proceed with API call
    console.log('Form submitted:', properties);

    try {
      // Extract features array from properties
      const features = [
        properties.f1_total_surface_area,
        properties.f2_non_polar_exposed_area,
        properties.f3_fractional_area_exposed_non_polar_residue,
        properties.f4_fractional_area_exposed_non_polar_part_residue,
        properties.f5_molecular_mass_weighted_exposed_area,
        properties.f6_average_deviation_from_standard_exposed_area,
        properties.f7_euclidian_distance,
        properties.f8_secondary_structure_penalty,
        properties.f9_spatial_distribution_constraints,
      ];

      // Call the API
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result = await response.json();
      console.log('API Response:', result);

      // Extract predictions from API response
      const garPrediction = result.predictions.GAR;
      const garExpPrediction = result.predictions['GAR-EXP'];
      const maePrediction = result.predictions.MAE;

      if (onPredict) {
        onPredict(garPrediction, garExpPrediction, maePrediction);
      }
    } catch (error) {
      console.error('Error calling prediction API:', error);
      alert(`Failed to get predictions: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <div className="protein-form-container">
      <h2 className="form-title">Protein Properties Input</h2>
      <form onSubmit={handleSubmit}>
        <div className="protein-grid">
          {(Object.keys(properties) as Array<keyof ProteinProperties>).map((key) => (
            <div key={key} className="form-group">
              <label htmlFor={key} className="form-label">
                {propertyLabels[key]}
              </label>
              <input
                type="number"
                id={key}
                name={key}
                min={0}
                step="any"
                value={properties[key] ? properties[key]!.toString() : ''}
                onChange={handleChange}
                className={`form-input ${errors[key] ? 'input-error' : ''}`}
                required
              />
              {errors[key] && (
                <span className="error-message">Must be greater than 0</span>
              )}
            </div>
          ))}
        </div>
        <button type="submit" className="submit-btn">
          Analyze Protein
        </button>
      </form>
    </div>
  );
};
