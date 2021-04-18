#ifndef SWEEP_RENORM_H
#define SWEEP_RENORM_H

namespace ctns{

template <typename Km>
void renorm_onedot(const directed_bond& dbond,
		   const comb<Km>& icomb, 
	           oper_dict<typename Km::dtype>& cqops,
	           oper_dict<typename Km::dtype>& lqops,
	           oper_dict<typename Km::dtype>& rqops,	
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   std::cout << "ctns::renorm_onedot" << std::endl;
   const auto& p = dbond.p;
   const auto& forward = dbond.forward;
   const auto& cturn = dbond.cturn;
   oper_dict<typename Km::dtype> qops;
   if(forward){
      if(!cturn){
	 oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
      }else{                                                  
	 oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
      }
      std::string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
      std::string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}

template <typename Km>
void renorm_twodot(const directed_bond& dbond,
		   const comb<Km>& icomb, 
	           oper_dict<typename Km::dtype>& c1qops,
	           oper_dict<typename Km::dtype>& c2qops,
	           oper_dict<typename Km::dtype>& lqops,
	           oper_dict<typename Km::dtype>& rqops,	
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   std::cout << "ctns::renorm_twodot" << std::endl;
   const auto& p = dbond.p;
   const auto& forward = dbond.forward;
   const auto& cturn = dbond.cturn;
   oper_dict<typename Km::dtype> qops;
   if(forward){
      if(!cturn){
	 oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, c1qops, qops);
      }else{
	 oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
      }
      std::string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      if(!cturn){
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c2qops, rqops, qops);
      }else{
	 //     
	 //        c2      
	 //        |
	 //   c1---p
	 //        |
	 //    l---*---r
	 //
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c1qops, c2qops, qops);
      }
      std::string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}

} // ctns

#endif
