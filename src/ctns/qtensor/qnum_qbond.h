#ifndef QNUM_QBOND_H
#define QNUM_QBOND_H

namespace ctns{

const int maxdim_per_sym = 65536; // 2**16
extern const int maxdim_per_sym;

// qbond: std::vector<std::pair<qsym,int>> dims;
class qbond{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dims;
      }
   public:
      // constructor
      qbond(){};
      qbond(const std::vector<std::pair<qsym,int>>& ds): dims(ds) {}
      // helpers
      int size() const{ return dims.size(); }
      qsym get_sym(const int i) const{ return dims[i].first; } 
      int get_dim(const int i) const{ return dims[i].second; }
      int get_parity(const int i) const{ return dims[i].first.parity(); }
      // total dimension
      int get_dimAll() const{
         int dim = 0;
         for(const auto& p : dims) dim += p.second;
         return dim;
      }
      // offset 
      std::vector<int> get_offset() const{
         std::vector<int> offset;
         int ioff = 0;
	 for(int i=0; i<dims.size(); i++){
	    offset.push_back(ioff);
	    ioff += dims[i].second; 
	 }
	 return offset;
      }
      // comparison
      bool operator ==(const qbond& qs) const{
	 bool ifeq = (dims.size() == qs.size());
	 if(not ifeq) return false;
	 for(int i=0; i<dims.size(); i++){
	    ifeq = ifeq && dims[i].first == qs.dims[i].first &&
		           dims[i].second == qs.dims[i].second;
	    if(not ifeq) return false;
	 }
	 return true;
      }
      void print(const std::string name, const bool debug=true) const{
	 std::cout << " qbond: " << name 
		   << " nsym=" << dims.size()
      	           << " dimAll=" << get_dimAll() 
		   << std::endl;
         // loop over symmetry sectors
	 if(debug){
            for(int i=0; i<dims.size(); i++){
               auto sym = dims[i].first;
               auto dim = dims[i].second;
	       std::cout << " " << sym << ":" << dim;
            }
	    std::cout << std::endl;
	 }
      }
   public:
      std::vector<std::pair<qsym,int>> dims;
};

} // ctns

#endif
