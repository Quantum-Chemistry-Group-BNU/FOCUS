#ifndef OPER_STACK_H
#define OPER_STACK_H

#include <thread>
#include "oper_dict.h"
#include "oper_io.h"

namespace ctns{

    // pool for mananging operators
    template <typename Tm>
        struct oper_pool{
            public:
                // constuctor
                oper_pool(const int _iomode,
                        const int _ioasync,
                        const bool _debug){
                    iomode = _iomode;
                    ioasync = _ioasync;
                    debug = _debug;
                }
                const oper_dict<Tm>& operator()(const std::string fqop) const{
                    return qstore.at(fqop);
                } 
                oper_dict<Tm>& operator()(const std::string fqop){
                    return qstore[fqop];
                }
                void display(const std::string msg) const{
                    std::cout << "qstore";
                    if(msg.size()>0) std::cout << "[" << msg << "]";
                    std::cout << ": size=" << qstore.size() << std::endl; 
                    for(const auto& pr : qstore){
                        std::cout << " fqop=" << pr.first << std::endl;
                    }
                }
                // fetch qops from memory / disk
                void fetch(const std::vector<std::string>& fneed);
                void save(const std::string frop){
                    //std::cout << "~~lzdX0~~ ioasync=" << ioasync << std::endl;
                    if(thrd.joinable()) thrd.join();
                    //std::cout << "~~lzdX1~~" << std::endl;
                    if(ioasync == 0){
                        ctns::oper_save<Tm>(iomode, frop, qstore.at(frop), debug);
                    }else if(ioasync == 1){
                        thrd = std::thread(&ctns::oper_save<Tm>, iomode, frop,
                                std::cref(qstore.at(frop)), debug);
                    }
                    //std::cout << "~~lzdX2~~" << std::endl;
                    fkept = frop;
                    //std::cout << "~~lzdX3~~" << std::endl;
                }
                void clean_up(){
                    // just add if to be compatible with serial version
                    if(thrd.joinable()) thrd.join();
                    fkept.clear();
                    // must be first join then clear, otherwise IO is not finished!
                    qstore.clear();
                }
                size_t size() const{
                    size_t sz = 0;
                    for(const auto& pr : qstore){
                        sz += pr.second.size();
                    }
                    return sz;
                }
            public:
                int iomode, ioasync;
                bool debug = false;
                std::map<std::string,oper_dict<Tm>> qstore; // for global storage
                std::string fkept;
                std::thread thrd;
        };

    template <typename Tm>
        void oper_pool<Tm>::fetch(const std::vector<std::string>& fneed){
            auto t0 = tools::get_time();
            if(debug_oper_io and debug){
                std::cout << "ctns::oper_pool<Tm>::fetch" << std::endl;
                std::cout << "fneed: size=" << fneed.size() << std::endl;
                for(const auto& fqop : fneed){
                    std::cout << " fqop=" << fqop << std::endl;
                }
                this->display("in");
            }
            // first release uncessary qops
            std::vector<std::string> frelease;
            for(auto& pr : qstore){
                auto& fqop = pr.first;
                if(fqop == fkept) continue; // keep fkept in memory until IO is finished
                auto result = std::find(fneed.begin(), fneed.end(), fqop);
                if(result == fneed.end()){
                    frelease.push_back(fqop);
                }
            }
            for(const auto& fqop : frelease){
                qstore.erase(fqop);
            }
            if(debug_oper_io and debug){
                std::cout << "frelease: size=" << frelease.size() << std::endl; 
                for(const auto& fqop : frelease){
                    std::cout << " fqop=" << fqop << std::endl;
                }
            }
            // then load new data is in memory
            for(const auto& fqop : fneed){
                if(qstore.find(fqop) != qstore.end()) continue;
                oper_load(iomode, fqop, qstore[fqop], debug);
            }
            if(debug_oper_io and debug){
                this->display("out");
                auto t1 = tools::get_time();
                tools::timing("ctns::oper_pool<Tm>::fetch", t0, t1);
            }
        }

} // ctns

#endif
