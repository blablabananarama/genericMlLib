use pyo3::prelude::*;
use pyo3::PyNumberProtocol;
use pyo3::wrap_pyfunction;

#[pyclass]
#[derive(Clone, Copy)]
struct Var {
    value: f32,
    grad: f32,

}

#[pymethods]
impl Var {
    #[new]
    fn new(value: f32, grad: f32 ) -> Self {
        Var { value, grad }
    }
    #[getter]
    fn value(&self) -> PyResult<f32> {
        Ok(self.value)
    }
}

#[pyproto]
impl PyNumberProtocol for Var {
    fn __add__(lhs: Var, rhs: Var) -> PyResult<Var> {
        Ok(Var{value: lhs.value + rhs.value, grad: lhs.grad}) 
    }
    fn __mul__(lhs: Var, rhs: Var) -> PyResult<Var> {
        Ok(Var{value: lhs.value * rhs.value, grad: lhs.grad}) 
    }
}


#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}


#[pymodule]
fn librust_grad(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Var>()?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

