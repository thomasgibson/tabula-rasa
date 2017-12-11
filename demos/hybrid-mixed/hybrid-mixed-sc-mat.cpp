
#include <petsc.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <Eigen/Dense>
#define restrict __restrict


#include <immintrin.h>

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_00_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t24[4] ;
    static const double  t0[3][4]  = {{1.0, 0.0, 0.0}, 
    {0.0, 1.0, 0.0}, 
    {0.0, 0.0, 1.0}};
    static const double  t1[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t2  = (-1 * coords[0][0]);
    double  t3  = (t2 + coords[1][0]);
    double  t4  = (t2 + coords[2][0]);
    double  t5  = ((t1[facet[0]][0][0] * t3) + (t1[facet[0]][1][0] * t4));
    double  t6  = (-1 * coords[0][1]);
    double  t7  = (t6 + coords[1][1]);
    double  t8  = (t6 + coords[2][1]);
    double  t9  = ((t1[facet[0]][0][0] * t7) + (t1[facet[0]][1][0] * t8));
    double  t10  = sqrt((t5 * t5) + (t9 * t9));
    static const double  t11[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t12  = (1 / ((t3 * t8) + (-1 * (t4 * t7))));
    double  t13  = (t8 * t12);
    double  t14  = ((t11[facet[0]][0] * t13) + (t11[facet[0]][1] * ((-1 * t7) * t12)));
    double  t15  = (t3 * t12);
    double  t16  = ((t11[facet[0]][0] * ((-1 * t4) * t12)) + (t11[facet[0]][1] * t15));
    double  t17  = (1 / sqrt((t14 * t14) + (t16 * t16)));
    double  t18  = (t14 * t17);
    double  t19  = (t16 * t17);
    double  t20  = (t10 * ((t18 * (t4 * t12)) + (t19 * t13)));
    static const double  t21[3][1][4]  = {{{0.5, -0.5, -0.5}}, 
    {{0.5, -0.5, -0.5}}, 
    {{0.0, 0.0, -1.0}}};
    double  t22  = (t10 * ((t18 * t15) + (t19 * (t7 * t12))));
    static const double  t23[3][1][4]  = {{{0.5, 0.5, 0.5}}, 
    {{0.0, 1.0, 0.0}}, 
    {{0.5, 0.5, 0.5}}};
    
    for (int  k  = 0; k < 3; k += 1)
    {
      t24[k] = (t23[facet[0]][0][k] * t22) + (t21[facet[0]][0][k] * t20);
      
    }
    
    for (int  j  = 0; j < 3; j += 1)
    {
      
      for (int  k  = 0; k < 3; k += 1)
      {
        #pragma coffee expression
        A(j, k) += t24[k] * t0[facet[0]][j];
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_00_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t24[4] ;
    static const double  t0[3][4]  = {{1.0, 0.0, 0.0}, 
    {0.0, 1.0, 0.0}, 
    {0.0, 0.0, 1.0}};
    static const double  t1[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t2  = (-1 * coords[0][0]);
    double  t3  = (t2 + coords[1][0]);
    double  t4  = (t2 + coords[2][0]);
    double  t5  = ((t1[facet[0]][0][0] * t3) + (t1[facet[0]][1][0] * t4));
    double  t6  = (-1 * coords[0][1]);
    double  t7  = (t6 + coords[1][1]);
    double  t8  = (t6 + coords[2][1]);
    double  t9  = ((t1[facet[0]][0][0] * t7) + (t1[facet[0]][1][0] * t8));
    double  t10  = sqrt((t5 * t5) + (t9 * t9));
    static const double  t11[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t12  = (1 / ((t3 * t8) + (-1 * (t4 * t7))));
    double  t13  = (t8 * t12);
    double  t14  = ((t11[facet[0]][0] * t13) + (t11[facet[0]][1] * ((-1 * t7) * t12)));
    double  t15  = (t3 * t12);
    double  t16  = ((t11[facet[0]][0] * ((-1 * t4) * t12)) + (t11[facet[0]][1] * t15));
    double  t17  = (1 / sqrt((t14 * t14) + (t16 * t16)));
    double  t18  = (t14 * t17);
    double  t19  = (t16 * t17);
    double  t20  = (t10 * ((t18 * (t4 * t12)) + (t19 * t13)));
    static const double  t21[3][1][4]  = {{{0.5, -0.5, -0.5}}, 
    {{0.5, -0.5, -0.5}}, 
    {{0.0, 0.0, -1.0}}};
    double  t22  = (t10 * ((t18 * t15) + (t19 * (t7 * t12))));
    static const double  t23[3][1][4]  = {{{0.5, 0.5, 0.5}}, 
    {{0.0, 1.0, 0.0}}, 
    {{0.5, 0.5, 0.5}}};
    
    for (int  k  = 0; k < 3; k += 1)
    {
      t24[k] = (t23[facet[0]][0][k] * t22) + (t21[facet[0]][0][k] * t20);
      
    }
    
    for (int  j  = 0; j < 3; j += 1)
    {
      
      for (int  k  = 0; k < 3; k += 1)
      {
        #pragma coffee expression
        A(j, k) += t24[k] * t0[facet[0]][j];
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel1_cell_to_00_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const double *const restrict *restrict w_0 , const double *const restrict *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][4]  = {{0.166666666666666, -0.166666666666667, -0.833333333333333}, 
    {0.666666666666666, -0.666666666666666, -0.333333333333333}, 
    {0.166666666666666, -0.166666666666667, -0.833333333333333}};
    static const double  t1[3][4]  = {{0.166666666666667, 0.833333333333333, 0.166666666666667}, 
    {0.166666666666667, 0.833333333333333, 0.166666666666667}, 
    {0.666666666666667, 0.333333333333333, 0.666666666666667}};
    double  t2  = (-1 * coords[0][0]);
    double  t3  = (t2 + coords[1][0]);
    double  t4  = (-1 * coords[0][1]);
    double  t5  = (t4 + coords[2][1]);
    double  t6  = (t2 + coords[2][0]);
    double  t7  = (t4 + coords[1][1]);
    double  t8  = ((t3 * t5) + (-1 * (t6 * t7)));
    double  t9  = (1 / t8);
    double  t10  = (t3 * t9);
    double  t11  = (t6 * t9);
    double  t12  = (t7 * t9);
    double  t13  = (t5 * t9);
    double  t14  = ((w_0[0][0] * (t10 * t11)) + (w_0[0][0] * (t12 * t13)));
    double  t15  = ((w_0[0][0] * (t11 * t11)) + (w_0[0][0] * (t13 * t13)));
    double  t16  = ((w_0[0][0] * (t10 * t10)) + (w_0[0][0] * (t12 * t12)));
    double  t17  = ((w_0[0][0] * (t11 * t10)) + (w_0[0][0] * (t13 * t12)));
    double  t18  = fabs(t8);
    static const double  t19[4]  = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    
    for (int  ip  = 0; ip < 3; ip += 1)
    {
      double  t25[4] ;
      double  t26[4] ;
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t17);
      double  t22  = (t20 * t16);
      double  t23  = (t20 * t15);
      double  t24  = (t20 * t14);
      
      for (int  k  = 0; k < 3; k += 1)
      {
        t25[k] = (t1[ip][k] * t24) + (t0[ip][k] * t23);
        t26[k] = (t1[ip][k] * t22) + (t0[ip][k] * t21);
        
      }
      
      for (int  j  = 0; j < 3; j += 1)
      {
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += (t26[k] * t1[ip][j]) + (t25[k] * t0[ip][j]);
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel1_cell_to_01_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const double *const restrict *restrict w_0 , const double *const restrict *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t13[1] ;
    static const double  t0[4]  = {1.0, -1.0, 1.0};
    double  t1  = (-1 * coords[0][0]);
    double  t2  = (t1 + coords[1][0]);
    double  t3  = (-1 * coords[0][1]);
    double  t4  = (t3 + coords[2][1]);
    double  t5  = (t1 + coords[2][0]);
    double  t6  = (t3 + coords[1][1]);
    double  t7  = ((t2 * t4) + (-1 * (t5 * t6)));
    double  t8  = (1 / t7);
    double  t9  = (t2 * t8);
    double  t10  = (t4 * t8);
    double  t11  = ((-1 * (0.5 * fabs(t7))) * ((((t9 * t10) + ((t5 * t8) * ((-1 * t6) * t8))) + ((t6 * t8) * ((-1 * t5) * t8))) + (t10 * t9)));
    static const double  t12[1]  = {1.0};
    
    for (int  k  = 0; k < 1; k += 1)
    {
      t13[k] = t12[k] * t11;
      
    }
    
    for (int  j  = 0; j < 3; j += 1)
    {
      
      for (int  k  = 0; k < 1; k += 1)
      {
        #pragma coffee expression
        A(j, k) += t0[j] * t13[k];
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel1_cell_to_10_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const double *const restrict *restrict w_0 , const double *const restrict *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t0  = (-1 * coords[0][0]);
    double  t1  = (t0 + coords[1][0]);
    double  t2  = (-1 * coords[0][1]);
    double  t3  = (t2 + coords[2][1]);
    double  t4  = (t0 + coords[2][0]);
    double  t5  = (t2 + coords[1][1]);
    double  t6  = ((t1 * t3) + (-1 * (t4 * t5)));
    double  t7  = (1 / t6);
    double  t8  = (t1 * t7);
    double  t9  = (t3 * t7);
    double  t10  = ((0.5 * fabs(t6)) * ((((t8 * t9) + ((t4 * t7) * ((-1 * t5) * t7))) + ((t5 * t7) * ((-1 * t4) * t7))) + (t9 * t8)));
    static const double  t11[1]  = {1.0};
    static const double  t12[4]  = {1.0, -1.0, 1.0};
    
    for (int  j  = 0; j < 1; j += 1)
    {
      double  t13  = (t11[j] * t10);
      
      for (int  k  = 0; k < 3; k += 1)
      {
        #pragma coffee expression
        A(j, k) += t12[k] * t13;
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel1_cell_to_11_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *const restrict *restrict coords , const double *const restrict *restrict w_0 , const double *const restrict *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[1]  = {1.0};
    double  t1  = (-1 * coords[0][0]);
    double  t2  = (-1 * coords[0][1]);
    double  t3  = (w_1[0][0] * (0.5 * fabs(((t1 + coords[1][0]) * (t2 + coords[2][1])) + (-1 * ((t1 + coords[2][0]) * (t2 + coords[1][1]))))));
    
    for (int  j  = 0; j < 1; j += 1)
    {
      
      for (int  k  = 0; k < 1; k += 1)
      {
        #pragma coffee expression
        A(j, k) += t3 * (t0[k] * t0[j]);
        
      }
      
    }
    
  }
  
}

 static inline void compile_slate (double  A2[3][3] , double **  coords , double **  w_0 , double **  w_1 , int8_t *  arg_cell_facets )
{
  int8_t (*cell_facets)[2] = (int8_t (*)[2])arg_cell_facets;
  /* Declare and initialize */
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor>  T0 ;
  T0.setZero();
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor>  T1 ;
  T1.setZero();
  /* Assemble local tensors */
  subkernel1_cell_to_00_cell_integral_otherwise(T1.block<3, 3>(0, 0), coords, w_0, w_1);
  subkernel1_cell_to_01_cell_integral_otherwise(T1.block<3, 1>(0, 3), coords, w_0, w_1);
  subkernel1_cell_to_10_cell_integral_otherwise(T1.block<1, 3>(3, 0), coords, w_0, w_1);
  subkernel1_cell_to_11_cell_integral_otherwise(T1.block<1, 1>(3, 3), coords, w_0, w_1);
  /* Loop over cell facets */
  
  for (unsigned int  i0  = 0; i0 < 3; i0 += 1)
  {
    if (cell_facets[i0][0] == 0) {
      if (cell_facets[i0][1] == 2) {
        subkernel0_exterior_facet_to_00_exterior_facet_integral_2(T0.block<3, 3>(0, 0), coords, &i0);
        
      }
       
    }
     if (cell_facets[i0][0] == 1) {
      subkernel0_interior_facet_to_00_exterior_facet_integral_otherwise(T0.block<3, 3>(0, 0), coords, &i0);
      
    }
     
  }
  /* Map eigen tensor into C struct */
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > T2((double *)A2);
  /* Linear algebra expression */
  T2 += T0 * (T1).inverse() * (T0).transpose();
  
}
            

extern "C" {
struct MapMask {
    /* Row pointer */
    PetscSection section;
    /* Indices */
    const PetscInt *indices;
};
struct EntityMask {
    PetscSection section;
    const int64_t *bottom;
    const int64_t *top;
};
static PetscErrorCode apply_extruded_mask(PetscSection section,
                                          const PetscInt mask_indices[],
                                          const int64_t mask,
                                          const int facet_offset,
                                          const int nbits,
                                          const int value_offset,
                                          PetscInt map[])
{
    PetscErrorCode ierr;
    PetscInt dof, off;
    /* Shortcircuit for interior cells */
    if (!mask) return 0;
    for (int bit = 0; bit < nbits; bit++) {
        if (mask & (1L<<bit)) {
            ierr = PetscSectionGetDof(section, bit, &dof); CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(section, bit, &off); CHKERRQ(ierr);
            for (int k = off; k < off + dof; k++) {
                map[mask_indices[k] + facet_offset] += value_offset;
            }
        }
    }
    return 0;
}
PetscErrorCode wrap_compile_slate(int start,
                      int end,
                      Mat arg0_0_, int32_t *arg0_0_map0_0, int32_t *arg0_0_map1_0, double *arg1_0, int32_t *arg1_0_map0_0, double *arg2_0, int32_t *arg2_0_map0_0, double *arg3_0, int32_t *arg3_0_map0_0, int8_t *arg4_0
                      ) {
  PetscErrorCode ierr;
  Mat arg0_0_0 = arg0_0_;
  double *arg1_0_vec[3];
    double *arg2_0_vec[1];
    double *arg3_0_vec[1];
  for ( int n = start; n < end; n++ ) {
    int32_t i = (int32_t)n;
    for (int i_0 = 0; i_0 < 3; i_0++) {
      arg1_0_vec[0 + i_0] = arg1_0 + (arg1_0_map0_0[i * 3 + i_0])* 2;
    };
    for (int i_0 = 0; i_0 < 1; i_0++) {
      arg2_0_vec[0 + i_0] = arg2_0 + (arg2_0_map0_0[i * 1 + i_0])* 1;
    };
    for (int i_0 = 0; i_0 < 1; i_0++) {
      arg3_0_vec[0 + i_0] = arg3_0 + (arg3_0_map0_0[i * 1 + i_0])* 1;
    };
    double buffer_arg0_0[3][3]  = {{0.0}};
    compile_slate(buffer_arg0_0, arg1_0_vec, arg2_0_vec, arg3_0_vec, arg4_0 + (i * 6));
                    {
    ierr = MatSetValuesLocal(arg0_0_0, 3, arg0_0_map0_0 + i * 3,
                                             3, arg0_0_map1_0 + i * 3,
                                             (const PetscScalar *)buffer_arg0_0,
                                             ADD_VALUES); CHKERRQ(ierr);
                    };
  }
  return 0;
}
}
        