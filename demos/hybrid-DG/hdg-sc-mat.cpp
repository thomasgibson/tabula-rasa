
#include <petsc.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <Eigen/Dense>
#define restrict __restrict


#include <immintrin.h>

template <typename Derived>
 static inline void subkernel0_cell_to_00_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const double *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t7[3][4] ;
    static const double  t0[6][4]  = {{0.109039009072877, 0.659027622374092, 0.231933368553031}, 
    {0.231933368553031, 0.659027622374092, 0.109039009072877}, 
    {0.109039009072877, 0.231933368553031, 0.659027622374092}, 
    {0.659027622374092, 0.231933368553031, 0.109039009072877}, 
    {0.231933368553031, 0.109039009072877, 0.659027622374092}, 
    {0.659027622374092, 0.109039009072877, 0.231933368553031}};
    double  t1  = (-1 * coords[0]);
    double  t2  = (-1 * coords[1]);
    double  t3  = fabs(((t1 + coords[2]) * (t2 + coords[5])) + (-1 * ((t1 + coords[4]) * (t2 + coords[3]))));
    static const double  t4[6]  = {0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333};
    
    for (int  j0  = 0; j0 < 3; j0 += 1)
    {
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        t7[j0][k0] = 0.0;
        
      }
      
    }
    
    for (int  ip  = 0; ip < 6; ip += 1)
    {
      double  t5  = ((t4[ip] * t3) * (((t0[ip][0] * w_0[0]) + (t0[ip][1] * w_0[1])) + (t0[ip][2] * w_0[2])));
      
      for (int  j0  = 0; j0 < 3; j0 += 1)
      {
        double  t6  = (t0[ip][j0] * t5);
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          t7[j0][k0] += t0[ip][k0] * t6;
          
        }
        
      }
      
    }
    
    for (int  j0  = 0; j0 < 3; j0 += 1)
    {
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        #pragma coffee expression
        A(j0*2+0, k0*2+0) += t7[j0][k0];
        #pragma coffee expression
        A(j0*2+1, k0*2+1) += t7[j0][k0];
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_cell_to_01_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const double *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t20[4] ;
    double  t21[4] ;
    double  t22[4] ;
    double  t23[4] ;
    static const double  t0[6][4]  = {{0.109039009072877, 0.659027622374092, 0.231933368553031}, 
    {0.231933368553031, 0.659027622374092, 0.109039009072877}, 
    {0.109039009072877, 0.231933368553031, 0.659027622374092}, 
    {0.659027622374092, 0.231933368553031, 0.109039009072877}, 
    {0.231933368553031, 0.109039009072877, 0.659027622374092}, 
    {0.659027622374092, 0.109039009072877, 0.231933368553031}};
    double  t1  = (-1 * coords[0]);
    double  t2  = (t1 + coords[2]);
    double  t3  = (-1 * coords[1]);
    double  t4  = (t3 + coords[5]);
    double  t5  = (t1 + coords[4]);
    double  t6  = (t3 + coords[3]);
    double  t7  = ((t2 * t4) + (-1 * (t5 * t6)));
    double  t8  = (1 / t7);
    double  t9  = (-1 * (t4 * t8));
    double  t10  = (-1 * ((-1 * t6) * t8));
    double  t11  = (-1 * ((-1 * t5) * t8));
    double  t12  = (-1 * (t2 * t8));
    double  t13  = fabs(t7);
    static const double  t14[6]  = {0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333};
    
    for (int  k  = 0; k < 3; k += 1)
    {
      t23[k] = 0.0;
      t22[k] = 0.0;
      t21[k] = 0.0;
      t20[k] = 0.0;
      
    }
    
    for (int  ip  = 0; ip < 6; ip += 1)
    {
      double  t15  = (t14[ip] * t13);
      double  t16  = (t12 * t15);
      double  t17  = (t11 * t15);
      double  t18  = (t10 * t15);
      double  t19  = (t9 * t15);
      
      for (int  k  = 0; k < 3; k += 1)
      {
        t20[k] += t0[ip][k] * t19;
        t21[k] += t0[ip][k] * t18;
        t22[k] += t0[ip][k] * t17;
        t23[k] += t0[ip][k] * t16;
        
      }
      
    }
    static const double  t24[4]  = {-1.0, 1.0, 0.0};
    static const double  t25[4]  = {-1.0, 0.0, 1.0};
    
    for (int  j0  = 0; j0 < 3; j0 += 1)
    {
      
      for (int  k  = 0; k < 3; k += 1)
      {
        #pragma coffee expression
        A(j0*2+0, k) += (t24[j0] * t20[k]) + (t25[j0] * t21[k]);
        #pragma coffee expression
        A(j0*2+1, k) += (t24[j0] * t22[k]) + (t25[j0] * t23[k]);
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_cell_to_10_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const double *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t20[4] ;
    double  t21[4] ;
    double  t22[4] ;
    double  t23[4] ;
    static const double  t0[6][4]  = {{0.109039009072877, 0.659027622374092, 0.231933368553031}, 
    {0.231933368553031, 0.659027622374092, 0.109039009072877}, 
    {0.109039009072877, 0.231933368553031, 0.659027622374092}, 
    {0.659027622374092, 0.231933368553031, 0.109039009072877}, 
    {0.231933368553031, 0.109039009072877, 0.659027622374092}, 
    {0.659027622374092, 0.109039009072877, 0.231933368553031}};
    double  t1  = (-1 * coords[0]);
    double  t2  = (t1 + coords[2]);
    double  t3  = (-1 * coords[1]);
    double  t4  = (t3 + coords[5]);
    double  t5  = (t1 + coords[4]);
    double  t6  = (t3 + coords[3]);
    double  t7  = ((t2 * t4) + (-1 * (t5 * t6)));
    double  t8  = (1 / t7);
    double  t9  = (-1 * (t4 * t8));
    double  t10  = (-1 * ((-1 * t6) * t8));
    double  t11  = (-1 * ((-1 * t5) * t8));
    double  t12  = (-1 * (t2 * t8));
    double  t13  = fabs(t7);
    static const double  t14[6]  = {0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333};
    
    for (int  k0  = 0; k0 < 3; k0 += 1)
    {
      t23[k0] = 0.0;
      t22[k0] = 0.0;
      t21[k0] = 0.0;
      t20[k0] = 0.0;
      
    }
    
    for (int  ip  = 0; ip < 6; ip += 1)
    {
      double  t15  = (t14[ip] * t13);
      double  t16  = (t12 * t15);
      double  t17  = (t11 * t15);
      double  t18  = (t10 * t15);
      double  t19  = (t9 * t15);
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        t20[k0] += t0[ip][k0] * t19;
        t21[k0] += t0[ip][k0] * t18;
        t22[k0] += t0[ip][k0] * t17;
        t23[k0] += t0[ip][k0] * t16;
        
      }
      
    }
    static const double  t24[4]  = {-1.0, 1.0, 0.0};
    static const double  t25[4]  = {-1.0, 0.0, 1.0};
    
    for (int  j  = 0; j < 3; j += 1)
    {
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        #pragma coffee expression
        A(j, k0*2+0) += (t24[j] * t20[k0]) + (t25[j] * t21[k0]);
        #pragma coffee expression
        A(j, k0*2+1) += (t24[j] * t22[k0]) + (t25[j] * t23[k0]);
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_cell_to_11_cell_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const double *restrict w_1 )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    double  t0  = (-1 * coords[0]);
    double  t1  = (-1 * coords[1]);
    double  t2  = fabs(((t0 + coords[2]) * (t1 + coords[5])) + (-1 * ((t0 + coords[4]) * (t1 + coords[3]))));
    static const double  t3[6]  = {0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333};
    static const double  t4[6][4]  = {{0.109039009072877, 0.659027622374092, 0.231933368553031}, 
    {0.231933368553031, 0.659027622374092, 0.109039009072877}, 
    {0.109039009072877, 0.231933368553031, 0.659027622374092}, 
    {0.659027622374092, 0.231933368553031, 0.109039009072877}, 
    {0.231933368553031, 0.109039009072877, 0.659027622374092}, 
    {0.659027622374092, 0.109039009072877, 0.231933368553031}};
    
    for (int  ip  = 0; ip < 6; ip += 1)
    {
      double  t6[4] ;
      double  t5  = ((((t4[ip][0] * w_1[0]) + (t4[ip][1] * w_1[1])) + (t4[ip][2] * w_1[2])) * (t3[ip] * t2));
      
      for (int  k  = 0; k < 3; k += 1)
      {
        t6[k] = t4[ip][k] * t5;
        
      }
      
      for (int  j  = 0; j < 3; j += 1)
      {
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t4[ip][j] * t6[k];
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_02_exterior_facet_integral_1 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  j0  = 0; j0 < 3; j0 += 1)
      {
        double  t23  = (t0[facet[0]][ip][j0] * t22);
        double  t24  = (t0[facet[0]][ip][j0] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j0*2+0, k) += t1[facet[0]][ip][k] * t24;
          #pragma coffee expression
          A(j0*2+1, k) += t1[facet[0]][ip][k] * t23;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_02_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  j0  = 0; j0 < 3; j0 += 1)
      {
        double  t23  = (t0[facet[0]][ip][j0] * t22);
        double  t24  = (t0[facet[0]][ip][j0] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j0*2+0, k) += t1[facet[0]][ip][k] * t24;
          #pragma coffee expression
          A(j0*2+1, k) += t1[facet[0]][ip][k] * t23;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_02_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  j0  = 0; j0 < 3; j0 += 1)
      {
        double  t23  = (t0[facet[0]][ip][j0] * t22);
        double  t24  = (t0[facet[0]][ip][j0] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j0*2+0, k) += t1[facet[0]][ip][k] * t24;
          #pragma coffee expression
          A(j0*2+1, k) += t1[facet[0]][ip][k] * t23;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_10_exterior_facet_integral_1 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t10 * t11);
    double  t13  = (t9 * t11);
    static const double  t14[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t15  = ((t14[facet[0]][0][0] * t3) + (t14[facet[0]][1][0] * t6));
    double  t16  = ((t14[facet[0]][0][0] * t7) + (t14[facet[0]][1][0] * t5));
    double  t17  = sqrt((t15 * t15) + (t16 * t16));
    static const double  t18[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t19  = (t18[ip] * t17);
      double  t20  = (t19 * t13);
      double  t21  = (t19 * t12);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        double  t23  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][k0] * t23;
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][k0] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_10_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t10 * t11);
    double  t13  = (t9 * t11);
    static const double  t14[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t15  = ((t14[facet[0]][0][0] * t3) + (t14[facet[0]][1][0] * t6));
    double  t16  = ((t14[facet[0]][0][0] * t7) + (t14[facet[0]][1][0] * t5));
    double  t17  = sqrt((t15 * t15) + (t16 * t16));
    static const double  t18[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t19  = (t18[ip] * t17);
      double  t20  = (t19 * t13);
      double  t21  = (t19 * t12);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        double  t23  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][k0] * t23;
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][k0] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_10_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t10 * t11);
    double  t13  = (t9 * t11);
    static const double  t14[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t15  = ((t14[facet[0]][0][0] * t3) + (t14[facet[0]][1][0] * t6));
    double  t16  = ((t14[facet[0]][0][0] * t7) + (t14[facet[0]][1][0] * t5));
    double  t17  = sqrt((t15 * t15) + (t16 * t16));
    static const double  t18[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t19  = (t18[ip] * t17);
      double  t20  = (t19 * t13);
      double  t21  = (t19 * t12);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        double  t23  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][k0] * t23;
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][k0] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_11_exterior_facet_integral_1 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = (((t12 * t12) * w_0[0]) + ((t13 * t13) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_11_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = (((t12 * t12) * w_0[0]) + ((t13 * t13) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_11_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = (((t12 * t12) * w_0[0]) + ((t13 * t13) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_12_exterior_facet_integral_1 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = ((((t13 * t13) * -1) * w_0[0]) + (((t14 * t14) * -1) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t1[facet[0]][ip][k] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_12_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = ((((t13 * t13) * -1) * w_0[0]) + (((t14 * t14) * -1) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t1[facet[0]][ip][k] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_12_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = ((((t13 * t13) * -1) * w_0[0]) + (((t14 * t14) * -1) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t1[facet[0]][ip][k] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_20_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t23[4] ;
      double  t24[4] ;
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        t23[k0] = t1[facet[0]][ip][k0] * t22;
        t24[k0] = t1[facet[0]][ip][k0] * t21;
        
      }
      
      for (int  j  = 0; j < 6; j += 1)
      {
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][j] * t24[k0];
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][j] * t23[k0];
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_21_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = (((t13 * t13) * w_0[0]) + ((t14 * t14) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t22[4] ;
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  k  = 0; k < 3; k += 1)
      {
        t22[k] = t1[facet[0]][ip][k] * t21;
        
      }
      
      for (int  j  = 0; j < 6; j += 1)
      {
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][j] * t22[k];
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_22_exterior_facet_integral_1 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t2  = (-1 * coords[0]);
    double  t3  = ((t1[facet[0]][0][0] * (t2 + coords[2])) + (t1[facet[0]][1][0] * (t2 + coords[4])));
    double  t4  = (-1 * coords[1]);
    double  t5  = ((t1[facet[0]][0][0] * (t4 + coords[3])) + (t1[facet[0]][1][0] * (t4 + coords[5])));
    double  t6  = sqrt((t3 * t3) + (t5 * t5));
    static const double  t7[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t8  = (t7[ip] * t6);
      
      for (int  j  = 0; j < 6; j += 1)
      {
        double  t9  = (t0[facet[0]][ip][j] * t8);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t9;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_exterior_facet_to_22_exterior_facet_integral_2 (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = ((((t12 * t12) * -1) * w_0[0]) + (((t13 * t13) * -1) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 6; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_02_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  j0  = 0; j0 < 3; j0 += 1)
      {
        double  t23  = (t0[facet[0]][ip][j0] * t22);
        double  t24  = (t0[facet[0]][ip][j0] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j0*2+0, k) += t1[facet[0]][ip][k] * t24;
          #pragma coffee expression
          A(j0*2+1, k) += t1[facet[0]][ip][k] * t23;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_10_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t10 * t11);
    double  t13  = (t9 * t11);
    static const double  t14[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t15  = ((t14[facet[0]][0][0] * t3) + (t14[facet[0]][1][0] * t6));
    double  t16  = ((t14[facet[0]][0][0] * t7) + (t14[facet[0]][1][0] * t5));
    double  t17  = sqrt((t15 * t15) + (t16 * t16));
    static const double  t18[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t19  = (t18[ip] * t17);
      double  t20  = (t19 * t13);
      double  t21  = (t19 * t12);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        double  t23  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][k0] * t23;
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][k0] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_11_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = (((t12 * t12) * w_0[0]) + ((t13 * t13) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_12_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t1[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = ((((t13 * t13) * -1) * w_0[0]) + (((t14 * t14) * -1) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  j  = 0; j < 3; j += 1)
      {
        double  t22  = (t0[facet[0]][ip][j] * t21);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t1[facet[0]][ip][k] * t22;
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_20_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t11 * t12);
    double  t14  = (t10 * t12);
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t4) + (t15[facet[0]][1][0] * t7));
    double  t17  = ((t15[facet[0]][0][0] * t8) + (t15[facet[0]][1][0] * t6));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t23[4] ;
      double  t24[4] ;
      double  t20  = (t19[ip] * t18);
      double  t21  = (t20 * t14);
      double  t22  = (t20 * t13);
      
      for (int  k0  = 0; k0 < 3; k0 += 1)
      {
        t23[k0] = t1[facet[0]][ip][k0] * t22;
        t24[k0] = t1[facet[0]][ip][k0] * t21;
        
      }
      
      for (int  j  = 0; j < 6; j += 1)
      {
        
        for (int  k0  = 0; k0 < 3; k0 += 1)
        {
          #pragma coffee expression
          A(j, k0*2+0) += t0[facet[0]][ip][j] * t24[k0];
          #pragma coffee expression
          A(j, k0*2+1) += t0[facet[0]][ip][j] * t23[k0];
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_21_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2][4]  = {{{0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.211324865405187, 0.788675134594813}}, 
    {{0.788675134594813, 0.0, 0.211324865405187}, 
    {0.211324865405187, 0.0, 0.788675134594813}}, 
    {{0.788675134594813, 0.211324865405187, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0}}};
    static const double  t2[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t3  = (-1 * coords[0]);
    double  t4  = (t3 + coords[2]);
    double  t5  = (-1 * coords[1]);
    double  t6  = (t5 + coords[5]);
    double  t7  = (t3 + coords[4]);
    double  t8  = (t5 + coords[3]);
    double  t9  = (1 / ((t4 * t6) + (-1 * (t7 * t8))));
    double  t10  = ((t2[facet[0]][0] * (t6 * t9)) + (t2[facet[0]][1] * ((-1 * t8) * t9)));
    double  t11  = ((t2[facet[0]][0] * ((-1 * t7) * t9)) + (t2[facet[0]][1] * (t4 * t9)));
    double  t12  = (1 / sqrt((t10 * t10) + (t11 * t11)));
    double  t13  = (t10 * t12);
    double  t14  = (t11 * t12);
    double  t15  = (((t13 * t13) * w_0[0]) + ((t14 * t14) * w_0[0]));
    static const double  t16[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t17  = ((t16[facet[0]][0][0] * t4) + (t16[facet[0]][1][0] * t7));
    double  t18  = ((t16[facet[0]][0][0] * t8) + (t16[facet[0]][1][0] * t6));
    double  t19  = sqrt((t17 * t17) + (t18 * t18));
    static const double  t20[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t22[4] ;
      double  t21  = ((t20[ip] * t19) * t15);
      
      for (int  k  = 0; k < 3; k += 1)
      {
        t22[k] = t1[facet[0]][ip][k] * t21;
        
      }
      
      for (int  j  = 0; j < 6; j += 1)
      {
        
        for (int  k  = 0; k < 3; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][j] * t22[k];
          
        }
        
      }
      
    }
    
  }
  
}

template <typename Derived>
 static inline void subkernel0_interior_facet_to_22_exterior_facet_integral_otherwise (const Eigen::MatrixBase<Derived> &  A_ , const double *restrict coords , const double *restrict w_0 , const unsigned int  facet[1] )
{
  {
    Eigen::MatrixBase<Derived> &  A  = const_cast<Eigen::MatrixBase<Derived> &>(A_);
    ;
    static const double  t0[3][2][6]  = {{{0.788675134594813, 0.211324865405187, 0.0, 0.0, 0.0, 0.0}, 
    {0.211324865405187, 0.788675134594813, 0.0, 0.0, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.788675134594813, 0.211324865405187, 0.0, 0.0}, 
    {0.0, 0.0, 0.211324865405187, 0.788675134594813, 0.0, 0.0}}, 
    {{0.0, 0.0, 0.0, 0.0, 0.788675134594813, 0.211324865405187}, 
    {0.0, 0.0, 0.0, 0.0, 0.211324865405187, 0.788675134594813}}};
    static const double  t1[3][2]  = {{1.0, 1.0}, 
    {-1.0, 0.0}, 
    {0.0, -1.0}};
    double  t2  = (-1 * coords[0]);
    double  t3  = (t2 + coords[2]);
    double  t4  = (-1 * coords[1]);
    double  t5  = (t4 + coords[5]);
    double  t6  = (t2 + coords[4]);
    double  t7  = (t4 + coords[3]);
    double  t8  = (1 / ((t3 * t5) + (-1 * (t6 * t7))));
    double  t9  = ((t1[facet[0]][0] * (t5 * t8)) + (t1[facet[0]][1] * ((-1 * t7) * t8)));
    double  t10  = ((t1[facet[0]][0] * ((-1 * t6) * t8)) + (t1[facet[0]][1] * (t3 * t8)));
    double  t11  = (1 / sqrt((t9 * t9) + (t10 * t10)));
    double  t12  = (t9 * t11);
    double  t13  = (t10 * t11);
    double  t14  = ((((t12 * t12) * -1) * w_0[0]) + (((t13 * t13) * -1) * w_0[0]));
    static const double  t15[3][2][1]  = {{{-1.0}, 
    {1.0}}, 
    {{0.0}, 
    {1.0}}, 
    {{1.0}, 
    {0.0}}};
    double  t16  = ((t15[facet[0]][0][0] * t3) + (t15[facet[0]][1][0] * t6));
    double  t17  = ((t15[facet[0]][0][0] * t7) + (t15[facet[0]][1][0] * t5));
    double  t18  = sqrt((t16 * t16) + (t17 * t17));
    static const double  t19[2]  = {0.5, 0.5};
    
    for (int  ip  = 0; ip < 2; ip += 1)
    {
      double  t20  = ((t19[ip] * t18) * t14);
      
      for (int  j  = 0; j < 6; j += 1)
      {
        double  t21  = (t0[facet[0]][ip][j] * t20);
        
        for (int  k  = 0; k < 6; k += 1)
        {
          #pragma coffee expression
          A(j, k) += t0[facet[0]][ip][k] * t21;
          
        }
        
      }
      
    }
    
  }
  
}

 static inline void compile_slate (double  A1[6][6] , double *  coords , double *  w_0 , double *  w_1 , double *  w_2 , int8_t *  arg_cell_facets )
{
  int8_t (*cell_facets)[2] = (int8_t (*)[2])arg_cell_facets;
  /* Declare and initialize */
  Eigen::Matrix<double, 15, 15, Eigen::RowMajor>  T0 ;
  T0.setZero();
  /* Assemble local tensors */
  subkernel0_cell_to_00_cell_integral_otherwise(T0.block<6, 6>(0, 0), coords, w_0, w_1);
  subkernel0_cell_to_01_cell_integral_otherwise(T0.block<6, 3>(0, 6), coords, w_0, w_1);
  subkernel0_cell_to_10_cell_integral_otherwise(T0.block<3, 6>(6, 0), coords, w_0, w_1);
  subkernel0_cell_to_11_cell_integral_otherwise(T0.block<3, 3>(6, 6), coords, w_0, w_1);
  /* Loop over cell facets */
  
  for (unsigned int  i0  = 0; i0 < 3; i0 += 1)
  {
    if (cell_facets[i0][0] == 0) {
      subkernel0_exterior_facet_to_02_exterior_facet_integral_otherwise(T0.block<6, 6>(0, 9), coords, w_2, &i0);
      subkernel0_exterior_facet_to_10_exterior_facet_integral_otherwise(T0.block<3, 6>(6, 0), coords, w_2, &i0);
      subkernel0_exterior_facet_to_11_exterior_facet_integral_otherwise(T0.block<3, 3>(6, 6), coords, w_2, &i0);
      subkernel0_exterior_facet_to_12_exterior_facet_integral_otherwise(T0.block<3, 6>(6, 9), coords, w_2, &i0);
      if (cell_facets[i0][1] == 1) {
        subkernel0_exterior_facet_to_02_exterior_facet_integral_1(T0.block<6, 6>(0, 9), coords, w_2, &i0);
        subkernel0_exterior_facet_to_10_exterior_facet_integral_1(T0.block<3, 6>(6, 0), coords, w_2, &i0);
        subkernel0_exterior_facet_to_11_exterior_facet_integral_1(T0.block<3, 3>(6, 6), coords, w_2, &i0);
        subkernel0_exterior_facet_to_12_exterior_facet_integral_1(T0.block<3, 6>(6, 9), coords, w_2, &i0);
        subkernel0_exterior_facet_to_22_exterior_facet_integral_1(T0.block<6, 6>(9, 9), coords, w_2, &i0);
        
      }
       if (cell_facets[i0][1] == 2) {
        subkernel0_exterior_facet_to_02_exterior_facet_integral_2(T0.block<6, 6>(0, 9), coords, w_2, &i0);
        subkernel0_exterior_facet_to_10_exterior_facet_integral_2(T0.block<3, 6>(6, 0), coords, w_2, &i0);
        subkernel0_exterior_facet_to_11_exterior_facet_integral_2(T0.block<3, 3>(6, 6), coords, w_2, &i0);
        subkernel0_exterior_facet_to_12_exterior_facet_integral_2(T0.block<3, 6>(6, 9), coords, w_2, &i0);
        subkernel0_exterior_facet_to_20_exterior_facet_integral_2(T0.block<6, 6>(9, 0), coords, w_2, &i0);
        subkernel0_exterior_facet_to_21_exterior_facet_integral_2(T0.block<6, 3>(9, 6), coords, w_2, &i0);
        subkernel0_exterior_facet_to_22_exterior_facet_integral_2(T0.block<6, 6>(9, 9), coords, w_2, &i0);
        
      }
       
    }
     if (cell_facets[i0][0] == 1) {
      subkernel0_interior_facet_to_02_exterior_facet_integral_otherwise(T0.block<6, 6>(0, 9), coords, w_2, &i0);
      subkernel0_interior_facet_to_10_exterior_facet_integral_otherwise(T0.block<3, 6>(6, 0), coords, w_2, &i0);
      subkernel0_interior_facet_to_11_exterior_facet_integral_otherwise(T0.block<3, 3>(6, 6), coords, w_2, &i0);
      subkernel0_interior_facet_to_12_exterior_facet_integral_otherwise(T0.block<3, 6>(6, 9), coords, w_2, &i0);
      subkernel0_interior_facet_to_20_exterior_facet_integral_otherwise(T0.block<6, 6>(9, 0), coords, w_2, &i0);
      subkernel0_interior_facet_to_21_exterior_facet_integral_otherwise(T0.block<6, 3>(9, 6), coords, w_2, &i0);
      subkernel0_interior_facet_to_22_exterior_facet_integral_otherwise(T0.block<6, 6>(9, 9), coords, w_2, &i0);
      
    }
     
  }
  /* Map eigen tensor into C struct */
  Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > T1((double *)A1);
  /* Linear algebra expression */
  T1 += ((T0).block<6, 6>(9, 9)) + (-((T0).block<6, 9>(9, 0)) * ((T0).block<9, 9>(0, 0)).inverse() * ((T0).block<9, 6>(0, 9)));
  
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
                      Mat arg0_0_, int32_t *arg0_0_map0_0, int32_t *arg0_0_map1_0, double *arg1_0, int32_t *arg1_0_map0_0, double *arg2_0, int32_t *arg2_0_map0_0, double *arg3_0, int32_t *arg3_0_map0_0, double *arg4_0, int8_t *arg5_0
                      ) {
  PetscErrorCode ierr;
  Mat arg0_0_0 = arg0_0_;
  for ( int n = start; n < end; n++ ) {
    int32_t i = (int32_t)n;
    double buffer_arg0_0[6][6] __attribute__((aligned(16))) = {{0.0}};
double buffer_arg1_0[6] __attribute__((aligned(16)));
double buffer_arg2_0[3] ;
double buffer_arg3_0[3] ;
    for (int i_0=0; i_0<3; ++i_0) {
buffer_arg1_0[i_0*2] = *(arg1_0 + (arg1_0_map0_0[i * 3 + i_0])* 2);
buffer_arg1_0[i_0*2 + 1] = *(arg1_0 + (arg1_0_map0_0[i * 3 + i_0])* 2 + 1);
};
for (int i_0=0; i_0<3; ++i_0) {
buffer_arg2_0[i_0*1] = *(arg2_0 + (arg2_0_map0_0[i * 3 + i_0])* 1);
};
for (int i_0=0; i_0<3; ++i_0) {
buffer_arg3_0[i_0*1] = *(arg3_0 + (arg3_0_map0_0[i * 3 + i_0])* 1);
}
    compile_slate(buffer_arg0_0, buffer_arg1_0, buffer_arg2_0, buffer_arg3_0, arg4_0, arg5_0 + (i * 6));
                    {
    ierr = MatSetValuesLocal(arg0_0_0, 6, arg0_0_map0_0 + i * 6,
                                             6, arg0_0_map1_0 + i * 6,
                                             (const PetscScalar *)buffer_arg0_0,
                                             ADD_VALUES); CHKERRQ(ierr);
                    };
  }
  return 0;
}
}
        